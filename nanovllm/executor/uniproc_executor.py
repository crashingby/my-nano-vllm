"""在调用方进程内创建单个 GPU Worker，并适配公共 Executor 协议。

UniProcExecutor 通过 WorkerWrapperBase 延迟构造 rank 0 的 Worker，依次初始化
设备与加载权重。普通 RPC 用单元素列表保留 rank 维度；execute_model() 和
sample_tokens() 使用 single_value=True，向调用方返回单个结果或单结果 Future。

所有 Worker 方法都在当前线程调用，不创建后台执行线程。non_block=True 时，
普通结果封装为已完成的 Future；AsyncModelRunnerOutput 封装为 AsyncOutputFuture，
由 result() 延后等待 GPU 到 CPU 的拷贝，使 CPU 调度可以与 GPU 工作重叠。
"""


from __future__ import annotations

import socket
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, TypeVar

from nanovllm.distributed import cleanup_dist_env_and_memory
from nanovllm.executor.abstract import Executor
from nanovllm.worker.worker_base import WorkerBase, WorkerWrapperBase
from nanovllm.logger import init_logger
from nanovllm.serial_utils import run_method
from nanovllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.outputs import ModelRunnerOutput,AsyncModelRunnerOutput

logger = init_logger(__name__)

class AsyncOutputFuture(Future):
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()

class UniProcExecutor(Executor):
    """管理一个本地 driver Worker，提供同步执行及 Future 形式的结果包装。

    Executor.get_class() 仅在 TP=1 时选择本类。唯一 Worker 的 rpc_rank 和
    global rank 均为 0，local_rank 则取配置中的 CUDA device index，缺省为 0。
    因而配置 cuda:1 时仍是一个 rank 0 Worker，只是绑定本机可见 GPU 1。

    构造时继承基类 __init__()，由其调用本类 _init_executor() 加载权重。
    正式 KV 仍需在 Engine 显存规划后通过 initialize_from_config() 初始化，
    再完成内核、Graph 和 sampler 的启动准备。

    Attributes:
        driver_worker: 同进程 WorkerWrapperBase 实例。显式生命周期方法在
            Wrapper 内选择参数和安装配置，其余属性查询转发给真实 GPU Worker。
    """
    def _init_executor(self) -> None:
        """创建 Wrapper 与 rank 0 Worker，依次初始化设备、加载权重。

        由 Executor.__init__() 调用。先保存 Wrapper，再取得会合地址和 rank
        参数，将单份构造字典包装为列表交给 init_worker()。Wrapper 在构造与
        init_device() 时安装配置，GPU Worker 在 load_model() 内安装模型构建上下文。

        Side Effects:
            保存 driver_worker，初始化设备与 world / TP group，采集显存基线、
            创建 Runner 并加载模型权重。返回后可查询 KV 规格和执行显存 profiling，
            正式 KV cache 尚未分配。Worker 构造、设备初始化或权重加载失败时，
            先关闭已创建的 Worker 并清理本进程分布式环境，再重新抛出原始异常。
        """
        # driver_worker 保存包装端点；两个编号均为 0，不由实际 CUDA device index 决定。
        self.driver_worker = WorkerWrapperBase(
            rpc_rank=0,
            global_rank=0,
        )
        # distributed_init_method 是 TCP 会合 URI，rank 是固定的通信身份 0，
        # local_rank 是进程可见的设备编号。单卡也建立分布式环境，供模型层查询分组。
        distributed_init_method, rank, local_rank = self._distributed_args()
        # kwargs 是唯一 Worker 的构造参数；is_driver_worker=True 表示承担结果返回职责。
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
        )
        try:
            # Wrapper 按 rpc_rank=0 从列表中选择 kwargs；此时 Worker 只保存公共状态。
            self.driver_worker.init_worker(all_kwargs=[kwargs])
            # 先绑定设备、建立通信组与显存基线，再创建 GPUModelRunner。
            self.driver_worker.init_device()
            # 模型加载由 Wrapper 转发给 GPU Worker，后者负责模型构建的配置上下文。
            self.driver_worker.load_model()
        except BaseException:
            # 构造未返回时，Engine 尚未取得 Executor，也没有注册退出回调。
            # 本地执行器仍持有 Wrapper，因此在这里先释放可能已创建的 Runner 资源。
            try:
                self.shutdown()
            except Exception:
                # 清理失败单独记录，保留显存预算不足等原始启动错误及其 traceback。
                logger.exception("Failed to shut down Worker after initialization failure")
            # 即使 Worker 清理失败，也继续销毁已经创建的 TP/WORLD/NCCL 进程组。
            try:
                cleanup_dist_env_and_memory()
            except Exception:
                logger.exception("Failed to clean up distributed state after initialization failure")
            raise

    # 保留原有的回环端口 helper；当前启动路径使用 network_utils.get_open_port()。
    @staticmethod
    def _get_open_port() -> int:
        """通过 IPv4 回环绑定查询临时空闲端口，保留原有本地 helper。

        当前 _distributed_args() 使用导入的 network_utils.get_open_port()，
        该网络工具会读取 VLLM_PORT；本静态方法固定在 127.0.0.1 上申请系统端口。

        Returns:
            操作系统实际分配的端口号，调用结束时探测 socket 已关闭。

        Side Effects:
            临时创建、绑定并关闭 TCP socket，不持有端口租约。关闭到真正创建
            PyTorch store 之间，其他进程仍可能占用该端口。
        """
        # socket_fd 绑定 port=0，让操作系统分配空闲端口，with 退出时立即关闭。
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_fd:
            socket_fd.bind(("127.0.0.1", 0))
            # getsockname() 返回实际监听地址，第二项是系统选择的端口。
            return int(socket_fd.getsockname()[1])

    def _distributed_args(self) -> tuple[str, int, int]:
        """生成 TCP 会合地址、全局 rank 和配置指定的本地 CUDA device index。

        get_ip() 优先读取 VLLM_HOST_IP，否则探测本机地址；get_open_port()
        使用 VLLM_PORT 指定的起点或系统分配端口。TCP URI 格式化负责 IPv6
        方括号，设备编号的解析只使用已由 DeviceConfig 规范化的 torch.device。

        Returns:
            (distributed_init_method, 0, local_rank)。配置 cuda 时 local_rank=0，
            配置 cuda:1 时 local_rank=1；这是当前进程可见的设备编号，受
            CUDA_VISIBLE_DEVICES 映射影响，不改变单 Worker 的全局 rank。

        Side Effects:
            网络工具可能短暂创建探测 socket，但不初始化 CUDA、创建 process group
            或长期占用端口。配置解析与端口绑定错误由对应工具向上传播。
        """

        # distributed_init_method 是将本机 IP 和当前可用端口格式化后的 TCP URI。
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # device_info 为 ["cuda"] 或 ["cuda", "N"]，保留与 vLLM 相同的解析步骤。
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        # 显式编号优先，未指定编号时用可见 GPU 0；可用性稍后由 Worker.init_device 检查。
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        # 单进程模型副本的通信身份始终为 0，不能用 local_rank 替代该值。
        return distributed_init_method, 0, local_rank
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        """在当前线程调用本地 Wrapper，并按两个标记包装 Worker 结果。

        non_block=False 时等待异步输出并返回 result 或 [result]；non_block=True
        时将异步输出交给 AsyncOutputFuture 延后回收，普通结果或调用异常则保存
        到已完成的 Future。两种路径均由 single_value 决定是否包装 rank 列表。

        Args:
            method: Wrapper / Worker 方法名，或接收 Wrapper 作为首参的 callable。
                run_method() 对字符串查询绑定方法，对 callable 额外传入端点。
            timeout: 保留公共 RPC 参数，当前不使用，不会中断已开始的本线程调用。
            args: 目标方法的位置参数，callable 的 Wrapper 首参由 run_method 补入。
            kwargs: 目标方法的关键字参数，None 时在本次调用中创建空字典。
            non_block: 为 True 时返回 Future，并延后异步输出的 CPU 回收；Worker
                的 Python 调用仍在当前线程执行，CUDA 工作可以继续异步运行。
            single_value: 为 True 时保留单个结果，为 False 时包装为单元素列表。
                即使 Worker 返回 None，也分别保持 None 与 [None] 的区别。

        Returns:
            同步为 result 或 [result]；Future 形式的 result() 返回同样的值。
            普通 KV 查询默认保留列表；执行与采样入口显式 single_value=True，
            直接向调用方返回单个结果或单结果 Future。

        Raises:
            NotImplementedError: 同步调用时找不到目标方法，由 run_method 抛出。
            Exception: 同步调用直接传播目标方法异常；Future 形式则将该异常
                对象保存到 Future，交给调用方通过 result() 取出。

        Side Effects:
            执行目标方法产生的副作用由 Wrapper / Worker 定义；Executor 本身
            只创建局部包装对象，不持有第二份模型输出或请求状态。
        """
        # 未传 kwargs 时创建本次调用的空参数字典，不使用跨调用共享的可变默认值。
        if kwargs is None:
            kwargs = {}
        # 同步分支不捕获目标方法异常，由当前调用栈直接向上报告。
        if not non_block:
            # result 是 Wrapper 适配或 Worker 执行后的原始返回值，可能为 None。
            result=run_method(self.driver_worker,method,args,kwargs)
            # 普通 RPC 保留外层 rank 列表，single_value 调用直接使用原结果。
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]
        # Future 分支也先在当前线程执行，只改变结果和异常的交付形式。
        try:
            result=run_method(self.driver_worker,method,args,kwargs)
            # 异步采样结果必须延后 get_output()，避免此处同步 D2H 阻塞下一轮调度。
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        # e 是目标调用抛出的原异常，保存同一个异常对象以保留其错误信息与 traceback。
        except Exception as e:
            future = Future[Any]()
            # set_exception() 同样将 Future 标记为完成，后续 result() 会重新抛出异常。
            future.set_exception(e)
        return future
    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """执行本轮调度，返回单个输出、待采样标记或其 Future 包装。

        覆盖基类的同步默认路径，直接使用 single_value=True。调用方取得 None
        时必须先调用 sample_tokens() 消费暂存 logits；使用 Future 时，先取
        result()，再根据其中的 None / ModelRunnerOutput 决定后续步骤。

        Args:
            scheduler_output: 本轮请求与 KV 更新、逐请求 token 执行量的完整快照。
            non_block: 是否以单结果 Future 交付返回值，执行本身仍在当前线程调用。

        Returns:
            当前 GPU Worker 对空调度返回空 ModelRunnerOutput，对成功 forward
            返回 None。non_block=True 时将相同的值保存在 Future 中，不增加
            rank 列表这一层包装。

        Raises:
            Exception: 同步执行错误直接传播；若 Future 已完成且失败，本方法也
                立即通过 result() 暴露错误，沿用 vLLM 的尽早失败处理。

        Side Effects:
            Worker 同步请求镜像；有调度 token 时准备输入、执行模型、写入 KV
            并暂存 logits。模式分派与 CUDA Graph 执行仍由 Runner 负责。
        """
        # output 是 Worker 的单个返回值或单结果 Future，此处不对它执行 output[0]。
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )

        # 仅检查已经完成的 Future，让 Python 执行错误立即暴露，不提前等待异步输出。
        if non_block and output.done():
            output.result()

        return output
    def sample_tokens(  # type: ignore[override]
        self,
        non_block: bool = False,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """消费最近一次 forward 的 logits，直接返回单个采样输出或其 Future。

        在 execute_model() 的实际结果为 None 后调用一次。该覆盖实现负责单结果
        包装，正常调用不进入基类 sample_tokens() 的列表取首项路径。

        Args:
            non_block: 是否返回单结果 Future。异步 Runner 输出经 AsyncOutputFuture
                延后等待 D2H；Worker 的 Python 调用仍在本线程完成。

        Returns:
            ModelRunnerOutput，或 result() 为该对象的 Future。输出包含请求 ID、
            请求行映射和有效 token 列表，未完成 prefill 的请求行为 []。

        Raises:
            RuntimeError: 同步调用时没有待采样状态，由 GPU Worker / Runner 抛出。
                Future 形式的该错误保存在 Future 内，调用 result() 时再抛出。

        Side Effects:
            Worker 消费执行状态、提交采样并更新 token 镜像；异步模式延后回收
            CPU 结果，Scheduler 的权威输出历史由外层消费结果时更新。
        """
        # single_value=True 在 collective_rpc 内完成单结果适配，保留 Future 本身。
        return self.collective_rpc(
            "sample_tokens",
            non_block=non_block,
            single_value=True,
        )
    def check_health(self) -> None:
        """沿用 vLLM 单进程执行器的空健康检查，不提交额外探测任务。

        当前没有 Worker 子进程或独立 RPC 连接需要轮询。执行错误由实际调用
        或 Future.result() 暴露；本方法直接返回，不主动同步 CUDA 设备。
        """
        return
    def shutdown(self) -> None:
        """委托本地 Wrapper 关闭 Worker，具体资源清理由 GPU Worker 组织。

        调用前应停止提交模型任务。Wrapper 允许真实 Worker 尚未构造成功时
        进入关闭入口；这里保留 driver_worker 引用，不另外管理全局通信组。

        Side Effects:
            GPU Worker 恢复 Python GC 扫描范围并委托 Runner 清理资源引用。
            其他引用或 allocator 仍可能保留存储，清理异常继续向上传播。
        """
        # worker 是已有 driver Wrapper 的局部引用，关闭请求再由其转发给真实 Worker。
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """声明支持通过异步输出回收，使 CPU 调度与 GPU 工作重叠。

        Worker 在本线程提交 forward 和 sampling，AsyncOutputFuture 延后等待
        D2H；Runner 通过 GPU token 回填准备下一轮输入，无需先取得 CPU token。
        """
        return True
