from collections import deque
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from multiprocessing.process import BaseProcess
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Lock as LockType
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from threading import Thread
from typing import Any, cast
import cloudpickle
import torch

from nanovllm.logger import  init_logger
from nanovllm.executor.abstract import Executor, FailureCallback
from nanovllm.config.vllm import NanovllmConfig
from nanovllm.envs import enable_envs_cache
from nanovllm.utils.system_utils import (
    _maybe_force_spawn,get_mp_context,
    set_process_title,
    decorate_logs,
)
from nanovllm.utils.torch_utils import (
    startup_omp_num_threads,
    set_torch_threads_for_runtime,
    OMP_NUM_THREADS_SET_BY_VLLM,
)
from nanovllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from nanovllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
import nanovllm.envs as envs
from nanovllm.platforms import current_platform
from nanovllm.distributed.parallel_state import (
    model_parallel_is_initialized,
    get_tp_group,
    destroy_model_parallel,
    destroy_distributed_environment
)

from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from nanovllm.worker.worker_base import WorkerWrapperBase
logger = init_logger(__name__)

class FutureWrapper(Future):
    """按 RPC 提交顺序回收响应，将结果或异常保存在对应的 Future 中。

    用于包装已提交、尚未回收响应的 RPC。多个调用共享同一个 futures_queue，
    各 Worker 的响应队列按执行顺序返回消息，因此回收时也必须保持提交顺序，
    否则后提交的 Future 可能把前一次调用的响应当成自己的结果。

    本类不创建后台线程；由调用 result() 的线程主动读取响应。Worker 已完成
    并不代表 Future 已完成；正常回收路径要写入结果或异常，done() 才变为 True。
    调用方应串行回收响应，共享 deque 和队列读取没有额外的并发消费保护。

    例如依次提交 F1、F2、F3 后，deque 从左到右为 [F3, F2, F1]。
    调用 F2.result() 会先回收 F1，再回收 F2；F3 留待后续回收。

    Attributes:
        futures_queue: Executor 内共享的待回收队列，左侧最新、右侧最早。
        get_response: 读取当前 RPC 所需 Worker 响应的无参回调，可能阻塞。
        aggregate: 将 get_response 的返回值整理为最终调用结果的回调。
    """

    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable[[Any], Any] = lambda x: x,
    ) -> None:
        """保存当前 RPC 的回收方式，并将自身加入共享待回收队列。

        Args:
            futures_queue: 同一 Executor 的 RPC 共用的 deque，按提交顺序登记。
            get_response: 延后执行的响应读取函数；构造时不会调用它。
            aggregate: 接收响应读取结果并返回最终结果，默认原样返回。

        Side Effects:
            初始化 Future 的未完成状态，并将自身加入 futures_queue 左端。
            不发送 RPC，也不启动线程或主动读取 Worker 的响应。
        """
        self.futures_queue: deque["FutureWrapper"] = futures_queue  # 共享同一个队列，不能复制。
        self.get_response: Callable[[], Any] = get_response  # 保存当前 RPC 对应的响应读取逻辑。
        self.aggregate: Callable[[Any], Any] = aggregate  # 保存结果整理方式，回收响应后再执行。
        super().__init__()  # 先建立 Future 的状态与内部同步机制，再让共享队列持有它。
        self.futures_queue.appendleft(self)  # 新提交的放左端，右端始终是最早登记的调用。

    def result(self, timeout: float | None = None) -> Any:
        """按顺序回收至当前调用，返回其缓存结果或抛出其缓存异常。

        Args:
            timeout: 当前只接受 None；本层不支持对整个顺序回收过程设置超时。
                消息读取自身是否超时，由 get_response 回调决定。

        Returns:
            response: 当前 RPC 经 aggregate 整理后的结果，类型由具体调用决定。
                当前 Future 已完成时，直接读取缓存，不会再次消费响应队列。

        Raises:
            RuntimeError: timeout 不为 None，或响应回调记录了 RuntimeError。
            CancelledError: 当前 Future 已被取消，由基类 result() 抛出。
            Exception: 当前 Future 保存的响应读取或结果整理异常，由基类重新抛出。

        Side Effects:
            从共享队列右端依次移除 Future，并回收其响应，直到当前 Future 完成。
            即使调用方尚未显式访问较早的 Future，它们也可能在此时被回收并完成。
        """
        if timeout is not None:
            # 基类的 timeout 无法覆盖前面多个 RPC 的回收，因此这里明确拒绝。
            raise RuntimeError("timeout not implemented")

        # 右端最早提交；只有把前面的响应交给各自 Future，才能读到本次的响应。
        while not self.done():
            future = self.futures_queue.pop()  # 当前对象也在队列中，回收到它后停止循环。
            future._wait_for_response()  # 保存被取出 Future 的结果或异常，推进其完成状态。
        return super().result()  # 由基类统一返回缓存结果，或抛出已保存的异常/取消状态。

    def _wait_for_response(self) -> None:
        """回收当前 RPC 的响应，整理后写入 Future 的结果或异常状态。

        由 result() 在当前对象从共享队列移除后调用；每个待回收项只消费一次。

        Side Effects:
            get_response 可能阻塞并消费 Worker 的响应队列；aggregate 随后处理结果。
            回调异常保存到当前 Future，交由其 result() 重新抛出。
            若 Future 已完成或被取消，则忽略写入终态时的 InvalidStateError，
            保留原有状态；这不代表共享队列支持多个线程同时回收。
        """
        try:
            response = self.aggregate(self.get_response())  # 先读取所需响应，再整理为最终结果。
            with suppress(InvalidStateError):
                self.set_result(response)  # 缓存成功结果，后续 result() 可重复读取。
        except Exception as e:
            # 读取和整理都可能失败；异常归属于当前 Future，延后到读取其结果时抛出。
            with suppress(InvalidStateError):
                self.set_exception(e)  # 仅忽略终态写入冲突，不吞掉正常记录的回调异常。


class MultiprocExecutor(Executor):
    """在 Executor 进程中组织本机 TP Worker 的启动与消息队列连接。

    每个 Worker 独占一个子进程。Executor 持有任务广播队列的写端，各 Worker
    持有它的读端；每个 Worker 另建自己的结果队列，Executor 连接对应读端。
    两个方向分开建队列，才能向全部 Worker 广播任务，并分别收取所需结果。

    当前按单机 TP 接入，因此 world_size == local_world_size，Worker 的
    global_rank == local_rank；没有跨节点 leader 选举或远端结果队列。
    WorkerProc 已包含子进程启动、队列连接、任务执行和输出回收逻辑，父进程已
    提供 RPC 分发与结果回收；进程监控及关闭接口仍待补齐，因此还不能判断端到端 TP
    已可运行。下面的初始化顺序描述父子进程应遵守的启动协议。

    Attributes:
        monitor_workers: 是否在收到全部 READY 后启动 Worker 存活监控。
        is_failed: 执行器是否进入不可继续接收任务的失败状态，初始为 False。
        failure_callback: 执行器失败时通知上层的回调，初始尚未注册。
        world_size: 一个模型副本内的 Worker 总数，当前等于 TP size。
        local_world_size: 本机 Worker 数，当前等于 world_size。
        rpc_broadcast_mq: Executor 写入、所有 Worker 读取的任务广播队列。
        workers: READY 后的进程句柄列表，按 Worker rank 排列。
        response_mqs: Executor 读取各 Worker 结果的队列，列表下标对应 rank。
        futures_queue: 已登记但尚未回收响应的 Future，左侧最新、右侧最早。
        output_rank: 对外提供模型输出的 Worker rank，当前单机纯 TP 时为 0。
    """

    # 声明当前没有流水线并行能力；下面的 pp_size / pcp_size 仅保留上游校验形式。
    supports_pp: bool = False

    def __init__(
        self,
        vllm_config: NanovllmConfig,
        monitor_workers: bool = True,
    ) -> None:
        """保存监控选项，再交给基类安装配置并调用本类初始化入口。

        Args:
            vllm_config: Engine 的完整配置，子进程启动参数和模型初始化共用它。
            monitor_workers: 是否启动进程存活监控；关闭后仍会等待启动 READY。

        Side Effects:
            基类保存各子配置后立即调用 _init_executor()，因此 monitor_workers
            必须在 super().__init__() 前设置，供启动路径读取。
        """
        self.monitor_workers: bool = monitor_workers  # 启动监控的开关，不控制 Worker 是否启动。
        super().__init__(vllm_config)  # 基类保存配置，再动态分派到下面的 _init_executor()。

    def _init_executor(self) -> None:
        """准备单机 TP 资源，启动 Worker，并按协议连接任务与结果队列。

        由 Executor.__init__() 调用。启动分为：准备线程数和会合地址、创建任务
        队列、启动子进程、接收 READY 与结果队列句柄、完成消息队列握手。
        READY 管道报告的是 Worker 初始化和句柄就绪，不能替代队列握手。

        Side Effects:
            设置进程环境与 Torch 线程数，创建共享内存、ZMQ 通道和子进程，
            保存 Worker / 消息队列 / Future 容器。进入进程创建的 try 块后若
            启动未完成，会关闭已登记 Worker 的死亡通知管道，并尝试终止它们。
            此阶段不下发 Engine 尚未规划的正式 KV cache 容量。

        Raises:
            AssertionError: 模型并行 world 与各并行维度乘积不一致，或本机
                Worker 上报后没有可连接的结果队列。
            Exception: 配置读取、资源创建、子进程初始化或队列握手失败。
        """
        # 一、准备执行器状态与进程启动环境；此时尚未创建任何 Worker。
        self._finalizer = weakref.finalize(self, self.shutdown)  # 登记执行器资源的关闭回调。
        self.is_failed: bool = False  # 监控发现故障后应设为 True，供 RPC 拒绝新任务。
        self.failure_callback: FailureCallback | None = None  # 上层稍后注册的故障通知入口。
        # 同时写入 world_size / local_world_size；当前 pp_size、pcp_size 固定为 1。
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        # 每个 Worker 必须能归入完整的模型并行网格；当前退化为 world_size == tp_size。
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )
        num_local_procs = self.local_world_size  # 本机即将启动的 Worker 数，不包含 Executor 自身。
        set_multiprocessing_worker_envs(num_local_procs)  # 按 Worker 数分配启动期 CPU 线程预算。
        # 各 Worker 共用的 PyTorch 分布式会合 URI；这是初始化地址，不是 RPC 队列地址。
        # get_open_port() 只探测当时可用的端口，返回后不继续占用端口。
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port(),
        )

        # 二、在父进程创建任务广播写端；单机唯一 Executor 直接承担创建职责。
        self.rpc_broadcast_mq: MessageQueue | None = None  
        # 传给子进程的连接信息，包含共享内存和 socket 地址；不携带每轮调度数据。
        scheduler_output_handle: Handle | None = None
        # 环境变量的 MB 数换算成字节，限制每个共享内存槽位容量；超大消息走队列的其他通道。
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        # 供队列需要建立远端 TCP 通道时使用；当前全为本地读者，实际通信使用共享内存 / IPC。
        mq_connect_ip = get_ip()
        logger.info(
            "mq_connect_ip=%s (local), "
            "world_size=%d, local_world_size=%d",
            mq_connect_ip,
            self.world_size,
            self.local_world_size,
        )
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,  # 总读者数：每个 Worker 都需要读到同一条任务。
            self.local_world_size,  # 本地读者数：当前等于总数，因此没有远端读者。
            max_chunk_bytes=max_chunk_bytes,  # 单槽容量，单位为字节，不是整个队列容量。
            connect_ip=mq_connect_ip,  # 队列地址配置，与分布式会合 URI 分开管理。
            enable_shm_tensor_arena=(self.parallel_config.enable_shm_tensor_arena),  # 可选 CPU Tensor arena。
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()  # Worker 据此连接广播读端。

        # 三、启动子进程；所有 Process、Pipe、Lock 使用兼容的 multiprocessing context。
        context = get_mp_context()  # 根据环境选择启动方式；WSL / CUDA 已初始化时会强制 spawn。
        # 传给全部 Worker 的同一把进程锁，预留给共享资源访问；当前 Worker 侧尚未接入使用。
        shared_worker_lock = context.Lock()
        # 累计保存已返回的启动句柄；READY 转换后仍保留这份列表，供后续启动失败时清理全部进程。
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False  # 整个 try 块成功后才置 True，之前的任何异常都触发 finally 清理。
        try:
            global_start_rank = 0  # 本机首个 Worker 的全局 rank；单机固定从 0 开始。
            # fork 会复制父进程已打开的管道 FD；记录它们，让后续子进程关闭不属于自己的端点。
            # 否则额外写端引用可能阻止 EOF，Worker 无法据死亡管道识别父进程已退出。
            # spawn 不复制这些任意 FD，所以用 None 跳过这套 fork 专用登记。
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )
            # 先启动所有 Worker 再集中等 READY，避免逐个等待卡住需要其他 rank 参与的初始化。
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank  # 模型副本内身份；当前等于本机编号。
                is_driver_worker = self._is_driver_worker(global_rank)  # 当前由 TP rank 0 承担 driver 职责。
                # 返回的是父进程管理句柄，proc.start() 已调用，但 Worker 不一定初始化完成。
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)  # 立刻登记，后续启动失败时仍能清理它。
                if inherited_fds is not None:
                    # 后续 fork 的 Worker 应关闭此前 Worker 的父端 FD，避免互相持有管道引用。
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # 四、接收全部 READY，按 rank 保存连接好结果读端的 WorkerProcHandle。
            # 子进程此时上报了结果队列句柄，仍需双方继续执行下一阶段队列握手。
            self.workers = WorkerProc.wait_for_ready(unready_workers)
            set_torch_threads_for_runtime()  # 调整父进程的稳态 CPU 算子线程数，不会远程修改子进程。
            if self.monitor_workers:
                self.start_worker_monitor()  # 预期开始监听进程退出；该方法目前尚未补齐。
            self.response_mqs: list[MessageQueue] = []  # Executor 读取结果的端点，按全局 rank 排列。
            # 当前所有 rank 都属于本机，下面的本地分支会覆盖全部 Worker。
            for rank in range(self.world_size):
                if rank < self.local_world_size:
                    # 队列由该 Worker 创建并写入；父进程持有的是连接后的读取端。
                    local_message_queue = self.workers[rank].worker_response_mq
                    # READY 转换必须已建立本地结果读端，否则不能进入正式 RPC 阶段。
                    assert local_message_queue is not None
                    self.response_mqs.append(local_message_queue)

            # 五、先完成任务广播队列握手，再完成各结果队列握手；Worker 侧须保持同样顺序。
            # 握手确认订阅已建立，避免刚启动就发送任务时丢失通知或消息。
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()  # 当前 Worker 的结果写端与父进程读端完成握手。
            self.futures_queue: deque[FutureWrapper] = deque[FutureWrapper]()  # 登记后续按序回收的 RPC。
            self._post_init_executor()  # 所有队列就绪后的扩展钩子，当前为空实现。
            success = True  # 只有进程、结果端点、队列握手和钩子全部成功，才跳过失败清理。
        finally:
            if not success:
                # 先关闭父进程持有的死亡管道写端，让已接入监控的 Worker 通过 EOF 请求退出。
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()  # 即使尚未收到 READY，也能通知已经启动的子进程。
                        uw.death_writer = None  # 记录该句柄已关闭，避免继续把它当成有效端点。
                # 对已登记子进程依次执行宽限等待、SIGTERM 和必要时的 SIGKILL。
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()  # 成功初始化后确定向 Engine 提供模型输出的 rank。

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        """保存总 Worker 数与本机 Worker 数，返回当前模型并行各维度的大小。

        Returns:
            (tp_size, pp_size, pcp_size): TP rank 数、PP stage 数和 prefill
                context parallel rank 数；后两项固定为 1，表示当前不启用。

        Side Effects:
            写入 world_size 和 local_world_size；当前限定全部 Worker 位于本机。
            查询可见 CUDA 设备数量；不创建进程组或启动子进程。

        Raises:
            ValueError: 本机可见 GPU 不足，或多 Worker 配置了固定 CUDA 设备编号。
        """
        self.world_size = self.parallel_config.world_size  # 一个模型副本所需的全部 Worker 数。
        self.local_world_size = self.world_size  # 单机场景全部在本机，无需按节点切分。
        # 在创建 MQ 和子进程前检查设备数量，避免部分 Worker 已启动后才发现缺卡。
        num_visible_devices = current_platform.device_count()
        if self.local_world_size > num_visible_devices:
            raise ValueError(
                f"tensor_parallel_size ({self.local_world_size}) exceeds the number "
                f"of visible CUDA devices ({num_visible_devices}) on this node"
            )
        # 多 Worker 分别绑定 local_rank；设备子集通过 CUDA_VISIBLE_DEVICES 指定。
        if self.local_world_size > 1 and self.device_config.device.index is not None:
            raise ValueError(
                "Multiprocess TP requires device='cuda' without an explicit index; "
                "select the visible GPU set with CUDA_VISIBLE_DEVICES"
            )
        tp_size = self.parallel_config.tensor_parallel_size  # 一次模型执行共同参与的 TP rank 数。
        pp_size = 1  # 不启用流水线并行，仅保留上游并行网格的表达形式。
        pcp_size = 1  # 不启用 prefill context parallelism，不会额外增加 Worker。
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        """提供所有 Worker 与队列就绪后的扩展位置，当前不执行额外操作。"""
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        """判断给定全局 rank 是否是所在 TP group 的第一个 Worker。

        Args:
            rank: 模型并行 world 内的 Worker 编号，由启动循环产生。

        Returns:
            is_driver_worker: rank 能被 TP size 整除时为 True；当前只有一个
                TP group，因此只有 rank 0 为 True，供 Worker 初始化选择职责。
        """
        return rank % self.parallel_config.tensor_parallel_size == 0
    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()
        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()
    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )
    def sample_tokens(  # type: ignore[override]
        self, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )
    def execute_dummy_batch(self) -> None:
        self.collective_rpc(
            "execute_dummy_batch",
            unique_reply_rank=self.output_rank,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable[..., Any],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """向全部 Worker 广播一次方法调用，返回结果或延后回收的 Future。

        方法调用总会广播到全部 TP rank，unique_reply_rank 只控制回包者。
        普通管理接口需要各 rank 的结果列表；模型执行接口可指定 output_rank，
        只取得一个模型输出。所有响应经共享 futures_queue 按提交顺序回收，
        不为每次调用创建独立的结果队列或后台读取线程。

        Args:
            method: WorkerWrapperBase 上的方法名，或可由 cloudpickle 序列化的
                callable。callable 在子进程中以 Wrapper 为首个参数，再接 args。
            timeout: 当前 RPC 读取响应的秒数预算，None 表示不限制等待。截止时间
                在广播前计算，多个结果队列共用剩余预算；non_block=True 时，
                延后调用 result() 也不会重新开始计时。发送队列未传入 timeout，
                因此不能保证整个方法在该时间内返回，也不会中断 Worker 的计算。
            args: 真实 Worker 方法或 callable 的位置参数，不含自动绑定的执行端点。
            kwargs: 调用的关键字参数，None 时使用空字典。
            non_block: 为 True 时广播后返回 Future，由 result() 主动读取响应；
                为 False 时在当前线程等待并返回结果。广播发送本身仍可能阻塞。
            unique_reply_rank: 唯一回包 Worker 的全局 rank，合法范围为
                [0, world_size)；None 表示所有 rank 均回包并返回 rank 有序列表。

        Returns:
            rpc_result: 同步模式下，未指定回包 rank 时是结果列表，指定时是该
                rank 的单个返回值；异步模式下是 Future，其 result() 返回同样
                形状的结果。Worker 成功返回 None 与读取响应失败含义不同。

        Raises:
            AssertionError: 任务广播队列尚未建立，不能提交 RPC。
            ValueError: unique_reply_rank 不在合法范围，广播前即拒绝。
            RuntimeError: 执行器已失败，或选中的回包 Worker 返回 FAILURE。
            TimeoutError: 读取响应超时。non_block=True 时，读取产生的异常在
                Future.result() 才抛出；序列化和发送阶段的异常仍立即传播。

        Side Effects:
            广播调用参数，登记待回收 Future，回收时消费所选 Worker 的结果队列。
            失败或超时不会自动撤回任务、重试或清空残留响应；错误处理应结束当前
            执行流程，不能假定仍可复用这些无调用 ID 的队列继续重试。
        """
        # 任务发送需要父进程持有广播写端；不能在初始化前或端点已清空后提交。
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")  # 已知 Worker 故障后，不再继续向其提交任务。
        # 必须先校验再广播；负数可索引 Python 列表，但没有对应的 Worker 会按负 rank 回包。
        if unique_reply_rank is not None and not 0 <= unique_reply_rank < self.world_size:
            raise ValueError(
                "unique_reply_rank must be in [0, world_size) or None, got "
                f"unique_reply_rank={unique_reply_rank}, world_size={self.world_size}"
            )
        deadline = None if timeout is None else time.monotonic() + timeout  # 共享截止时刻，使用单调时钟秒数。
        kwargs = kwargs or {}  # 统一成可直接传给 func(*args, **kwargs) 的字典。
        output_rank = unique_reply_rank  # 消息中的回包选择，Worker 在执行方法之后检查它。
        aggregate: Callable[[Any], Any] = lambda x: x  # 当前没有额外聚合器，保留 Future 的原样返回逻辑。
        if isinstance(method, str):
            send_method = method  # 子进程用 getattr(self.worker, method) 查找绑定方法。
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)  # 子进程还原并绑定端点。
        # 四元组格式与 WorkerProc._execute_worker_rpc() 完全对应；所有 Worker 各读到一次。
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
        response_mqs: Sequence[MessageQueue] = self.response_mqs  # 默认按 rank 读取全部结果队列。
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)  # rank 0 也有效，不能用 if output_rank 判断。

        def get_response() -> Any:
            """读取当前 RPC 所需的响应，按回包模式返回列表或单个结果。

            Returns:
                rpc_result: output_rank 为 None 时是 responses 列表，否则取
                    唯一队列的 responses[0]；这里不会主动等待其他 rank 的回包。

            Raises:
                TimeoutError: 某条结果队列在剩余等待预算内未返回响应。
                RuntimeError: 某个回包状态不是 SUCCESS，错误正文来自 Worker。

            Side Effects:
                消费闭包中 response_mqs 的下一条响应；只有轮到当前 Future
                被回收时才调用，避免与前一次 RPC 的结果错位。中途失败时不会
                继续读取剩余队列，也不会主动清理已经到达或后续到达的响应。
            """
            responses: list[Any] = []  # 按所选队列的 rank 顺序收集方法返回值。
            for mq in response_mqs:
                # 每个队列只拿剩余时间，不能为每个 rank 都重新获得完整 timeout。
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)  # 对应 Worker 封装的 (状态, payload)。
                except TimeoutError as e:
                    # 补上方法信息，便于定位是哪次 RPC 等待超时，并保留底层异常链。
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    # 失败 payload 是子进程的错误字符串，完整 traceback 查看 Worker 日志。
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)  # 只有成功响应才作为调用返回值收集，None 也保留。
            return responses[0] if output_rank is not None else responses

        # 即使同步调用也先登记 Future，以便先回收此前尚未消费的 RPC，维护共享队列顺序。
        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()  # 非阻塞只延后回收，未启动新的结果读取线程。

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]) -> None:
        """启动失败时等待已创建的子进程退出，必要时逐级发送终止信号。

        Args:
            worker_procs: 本次启动已登记的 Process 对象，可能包含已经退出的进程。

        Side Effects:
            先等待环境变量指定的宽限期，再对存活进程发送 SIGTERM，等待最多
            4 秒后对仍存活者发送 SIGKILL。本方法不 join() 回收进程句柄，
            也不在 SIGKILL 后继续等待；管道和消息队列的清理由调用方管理。
        """

        def wait_for_termination(procs: list[BaseProcess], timeout: float) -> bool:
            """轮询给定子进程是否全部退出，最长等待 timeout 秒。

            Args:
                procs: 本次等待涉及的进程快照，循环中反复检查其存活状态。
                timeout: 最长等待秒数；当前使用 time.time() 的墙钟差值计时。

            Returns:
                all_exited: 等待期内全部退出时为 True，超时后为 False；解释器
                    退出阶段 time 不可用时只检查一次，不继续等待。

            Side Effects:
                当前线程每轮最多休眠 0.1 秒，不向子进程发送信号。
            """
            if not time:
                # 解释器退出后期可能把模块全局变量替换为 None，此时不能再调用 time。
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()  # 本次等待阶段的开始时刻，单位为秒。
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)  # 给 Worker 留出释放设备、队列和 Python 对象的执行机会。
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]  # 每次重新筛选存活进程。
        initial_count = len(active_procs())  # 进入清理时仍需等待的进程数，仅用于日志。

        # 调用方已关闭死亡通知写端，先给 Worker 自行清理并退出的时间。
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        # 宽限期结束后重新取快照，只向仍存活的 Worker 请求终止。
        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()  # POSIX 下发送 SIGTERM，Worker 可通过信号处理器进入清理路径。
        if not wait_for_termination(active_procs(), 4):
            remaining = active_procs()  # SIGTERM 后再等 4 秒，筛出仍未退出的子进程。
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()  # POSIX 下发送不可捕获的 SIGKILL，不再依赖 Worker 执行清理代码。
    def shutdown(self):
        """Properly shut down the executor and its workers."""
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        logger.debug_once("[shutdown] Executor: complete")
    
    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return
    def _get_output_rank(self) -> int:
        """返回负责向 Engine 提供模型输出的 Worker rank。

        Returns:
            output_rank: 当前 world_size 等于 TP size，因此返回 0。保留上游
                world_size - tp_size 的形式，但不代表当前支持多个 PP stage。
        """
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
        )
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True






@dataclass
class UnreadyWorkerProcHandle:
    """父进程管理已启动、尚未完成 READY 协议的 Worker 所需的信息。

    此对象不是真实 Worker；它保存在 Executor 进程中，用于等待启动通知和
    在启动失败时清理子进程。READY 成功后转换成 WorkerProcHandle。

    Attributes:
        proc: 已调用 start() 的进程对象，不代表设备和模型已初始化完成。
        rank: Worker 在模型副本中的全局编号，用于保持结果列表的 rank 顺序。
        ready_pipe: 父进程持有的接收端，读取子进程的 READY 状态和结果队列句柄。
        death_writer: 父进程保留的死亡通知写端，关闭后子进程读端可观察到 EOF。
            None 表示该句柄已经关闭，或构造时未提供。
    """

    proc: BaseProcess  # 控制子进程生命周期，不通过它直接调用真实 Worker 的方法。
    rank: int  # 启动时分配的模型并行身份；当前也等于 local_rank。
    ready_pipe: Connection  # READY 的数据流方向是子进程写、父进程读。
    death_writer: Connection | None = None  # 保持此端打开，表示父进程尚未请求 Worker 退出。

@dataclass
class WorkerProcHandle:
    """父进程在收到 READY 并连接结果队列后持有的 Worker 管理端点。

    READY 管道完成使命后关闭，因此本对象不再保存 ready_pipe。任务和结果
    队列随后仍需执行 wait_until_ready()，不能仅凭这个对象已创建就发送任务。

    Attributes:
        proc: 原来的子进程对象，供监控与退出清理使用。
        rank: Worker 的全局 rank，转换句柄时保持不变。
        worker_response_mq: Executor 接收该 Worker 结果的本地队列读端；Worker
            在自己的进程中持有同一队列的写端。当前单机路径要求它不为 None。
        peer_worker_response_mqs: 上游跨节点结果队列端点列表，当前单机路径预期为空。
        death_writer: 从启动句柄接续持有的死亡通知写端，正常运行时保持打开。
    """

    proc: BaseProcess  # 继续管理同一子进程，不因 READY 转换重新创建进程。
    rank: int  # 用于结果队列下标和输出 rank 选择。
    worker_response_mq: MessageQueue | None  # 父进程的结果读端，不是任务广播队列。
    peer_worker_response_mqs: list[MessageQueue | None]  # 保留上游字段，单机无远端结果端点。
    death_writer: Connection | None = None  # 维持父进程存活 / 退出通知关系。

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        """复用启动句柄中的进程信息，补入已经连接的结果队列端点。

        Args:
            unready_handle: READY 转换前的父进程管理句柄。
            worker_response_mq: 根据子进程上报的 Handle 连接得到的本地结果读端。
            peer_worker_response_mqs: 连接得到的远端结果端点，当前单机预期为空。

        Returns:
            worker_proc_handle: 持有同一 proc、rank、death_writer，并附带结果
                队列的新句柄；不包含仅供启动握手使用的 ready_pipe。

        Side Effects:
            只构造父进程内的管理对象，不关闭旧句柄、不创建进程、不执行队列握手。
            新旧句柄中的进程及死亡通知写端是相同对象，清理时需注意共享引用。
        """
        return cls(
            proc=unready_handle.proc,  # 复用已经启动的 Process 对象。
            rank=unready_handle.rank,  # 保持 Worker 在模型副本内的通信身份。
            worker_response_mq=worker_response_mq,  # 接入 Executor 读取结果的本地端点。
            peer_worker_response_mqs=peer_worker_response_mqs,  # 单机不会额外接入远端端点。
            death_writer=unready_handle.death_writer,  # 持续持有，避免 READY 后误通知子进程退出。
        )

class WorkerProc:
    """组织 Worker 子进程的生命周期、RPC 执行以及同步或异步结果回传。

    同一个类的静态方法与实例方法在不同进程里使用。Executor 父进程调用
    make_worker_process() 启动子进程，再用 wait_for_ready() 连接结果读端；
    子进程从 worker_main() 进入，创建本类实例及其包装的真实 GPU Worker，
    上报 READY、完成队列握手后，才开始从任务队列接收并执行 RPC。

    任务广播队列由 Executor 创建，所有 TP Worker 各持一个读端；结果队列
    由各 Worker 分别创建，Executor 是每条结果队列唯一的读者。output_rank
    仅控制哪些 Worker 回包，全部 Worker 仍执行相同 RPC 并参与 TP 通信。

    异步调度时，模型输出先进入当前进程的线程队列，由输出线程等待设备输出
    可读并回包。CUDA 事件和设备 Tensor 保留在 Worker 内，进程间只传整理后
    的结果。父进程退出时，死亡管道的 EOF 触发队列取消，让任务主线程退出
    接收循环，再由 worker_main() 的 finally 清理真实 Worker 和分布式环境。

    Attributes:
        rank: 当前 Worker 在模型副本内的全局 rank，单机时等于 local_rank。
        worker: 当前进程内的 WorkerWrapperBase，负责访问真实 GPU Worker。
        rpc_broadcast_mq: 子进程接收任务的队列读端，关闭后置为 None。
        worker_response_mq: 当前 Worker 回传结果的队列写端，关闭后置为 None。
        peer_response_handles: READY 消息中保留的远端结果队列句柄，单机为空。
        use_async_scheduling: 是否将回包交给进程内的输出线程处理。
        async_output_queue: 异步模式才创建的 queue.Queue，按提交顺序保存待回收输出。
        async_output_copy_thread: 异步模式才创建的守护线程，执行输出回收与回包。
    """

    READY_STR = "READY"  # 子进程完成初始化后通过 READY 管道上报的成功状态。
    rpc_broadcast_mq: MessageQueue | None  # 子进程接收 Executor 广播任务的读端。
    worker_response_mq: MessageQueue | None  # 子进程向 Executor 返回结果的写端。

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: NanovllmConfig
    ) -> None:
        """在 Worker 子进程内连接任务读端，并创建本 Worker 的结果写端。

        在真实 Worker 构造、设备初始化和模型加载后调用。当前仅支持单机 TP，
        直接使用上游的本地队列路径，无需判断节点编号或建立跨节点广播通道。

        Args:
            input_shm_handle: Executor 的任务广播队列句柄，全部 Worker 共用；
                通过各自 rank 区分读者，确保每个 Worker 都能收到同一条任务。
            vllm_config: 保留与上游一致的调用签名，当前单机分支无需读取该配置。

        Side Effects:
            映射任务队列的共享内存并连接本地 socket，创建结果队列的共享内存
            和 socket。这里只连接或创建端点，不执行 wait_until_ready()：
            子进程须先通过 READY 管道上报结果队列 Handle，让父进程连接读端，
            然后双方按任务队列、结果队列的顺序握手，避免相互等待。
        """
        # 连接 Executor 已创建的广播队列；当前 Worker 是读者，不再创建任务写端。
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank,  # rank 对应广播队列中该 Worker 的读者身份。
        )
        # 结果队列由当前 Worker 写入，唯一读者是 Executor；读者数与 TP size 无关。
        self.worker_response_mq = MessageQueue(n_reader=1, n_local_reader=1)
        # 后续 READY 消息沿用上游字段；单机没有需要额外连接的远端结果队列。
        self.peer_response_handles: list[Handle] = []

    def __init__(
        self,
        vllm_config: NanovllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ) -> None:
        """在已启动的子进程中初始化真实 Worker、输出线程和消息队列端点。

        供 worker_main() 调用：先通过 Wrapper 构造 GPU Worker，再绑定设备、
        建立分布式环境并加载模型，随后按调度模式准备输出回收线程和消息队列。
        此构造函数不再创建 Worker 子进程，也不发送 READY 或执行队列握手；
        后两步由 worker_main() 在构造成功后组织。正式 KV cache 的容量与分配
        仍需等待 Engine 完成显存规划，再通过 RPC 下发。

        Args:
            vllm_config: 传入当前子进程的完整配置，供真实 Worker 构造与运行使用。
            local_rank: 本机 Worker 编号，也是 Wrapper 选择构造参数的 rpc_rank；
                真实 Worker 用它绑定当前进程可见的 CUDA 设备。
            rank: 一个模型副本内的全局 Worker 编号，用于分布式通信和结果返回
                身份；当前单机 TP 时等于 local_rank。
            distributed_init_method: 各 rank 共用的 PyTorch 分布式会合 URI，
                与传递执行命令的任务广播队列地址分别管理。
            input_shm_handle: Executor 创建的任务广播队列句柄，当前 Worker
                按自己的 rank 连接读端，不重复创建任务写端。
            shared_worker_lock: 父进程创建并传给全部 Worker 的同一把进程锁。
                Wrapper 会从构造参数中取出它；当前没有实际使用方，不传给
                GPU Worker 的构造函数。
            is_driver_worker: 是否承担 driver 职责；当前单机 TP 仅 rank 0 为 True。

        Side Effects:
            创建真实 Worker、CUDA / 分布式资源及模型权重，设置进程标题和日志
            身份；异步调度时额外启动当前进程内的输出线程。随后连接任务读端、
            创建结果写端，并启用本进程环境变量缓存。初始化异常向子进程入口
            传播；构造成功不代表已完成消息队列握手或可以开始接收任务。
        """
        self.rank: int = rank  # WorkerProc 自身也保存通信身份，后续 RPC 可据此判断是否回包。
        # Wrapper 是当前进程内真实 Worker 的访问入口，自身不创建进程或传输消息。
        # rpc_rank 选择构造参数，global_rank 选择后续下发的 KV 分配方案。
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # 沿用 Wrapper 接受参数列表的接口，只填入本 Worker 对应的一项。
        # 例如 TP=2、local_rank=1 时是 [{}, kwargs1]，不会在本进程初始化另一个 Worker。
        all_kwargs: list[dict[str, Any]] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,  # Wrapper 安装配置作用域，真实 Worker 保存各子配置引用。
            "local_rank": local_rank,  # 设备初始化时用于选择本机可见 GPU。
            "rank": rank,  # 分布式 world 内的身份，真实 Worker 构造时回写并行配置。
            "distributed_init_method": distributed_init_method,  # 各 rank 通过同一地址会合。
            "is_driver_worker": is_driver_worker,  # 显式传入当前 rank 的职责标记。
            "shared_worker_lock": shared_worker_lock,  # 由 Wrapper 取出，避免传入不接收它的 Worker。
        }
        wrapper.init_worker(all_kwargs)  # 这里只构造真实 Worker，设备和模型资源随后再初始化。
        self.worker: WorkerWrapperBase = wrapper  # 后续 init_device / load_model / RPC 共用这一端点。
        # 通信组尚未建立时先显示 Worker，让设备初始化阶段的日志也能标识进程来源。
        self.setup_proc_title_and_log_prefix(
            enable_ep=False  # 当前没有专家并行，不需要追加 EP rank。
        )
        self.worker.init_device()  # 绑定 CUDA 设备、建立 world / TP group 并创建 GPUModelRunner。
        # 通信组建立后已能查询 TP rank，更新为 Worker_TP0 等进程标题和日志身份。
        self.setup_proc_title_and_log_prefix(
            enable_ep=False
        )
        self.worker.load_model()  # 在当前设备上构造模型、加载本 rank 的权重；正式 KV 尚待规划。

        scheduler_config = vllm_config.scheduler_config  # 读取与 Engine 一致的异步调度开关。
        self.use_async_scheduling: bool = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            # 这是进程内的线程队列，暂存待回收输出；不是父子进程之间的结果消息队列。
            self.async_output_queue: queue.Queue[Any] = queue.Queue()
            # 输出线程等待 D2H 完成、整理 CPU 输出并回包，使任务主线程能继续处理后续调用。
            # CUDA 事件和设备 Tensor 留在当前 Worker 进程内，由回收完成后的结果跨进程传输。
            self.async_output_copy_thread: Thread = Thread(
                target=self.async_output_busy_loop,  # 循环从进程内队列取输出，调用 enqueue_output()。
                daemon=True,  # 作为当前 Worker 的辅助线程运行，不独立维持进程存活。
                name="WorkerAsyncOutputCopy",  # 用于调试时识别输出回收线程。
            )
            # 此时输出队列为空，线程先阻塞等待；正式任务循环要在 READY 和握手后才开始。
            self.async_output_copy_thread.start()

        # 当前 FA2 的 KV block size 已由 CacheConfig 校验，无需动态调整后端 block size。
        self._init_message_queues(input_shm_handle, vllm_config)  # 连接任务读端并创建结果写端，暂不握手。
        # 启动配置确定后缓存环境变量查询结果；此后再修改环境变量不会自动反映到 envs 属性。
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: NanovllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        """在父进程中创建两条启动管理管道，启动指定 rank 的 Worker 子进程。

        Args:
            vllm_config: 传给子进程的完整配置；spawn 要求这些启动参数可序列化。
            local_rank: 本机 Worker 编号，供子进程设备初始化选择对应 GPU。
            rank: 模型副本内的全局 Worker 编号；当前单机时等于 local_rank。
            distributed_init_method: 所有 Worker 共用的分布式进程组会合 URI。
            input_shm_handle: Executor 创建的任务广播队列句柄，子进程据此连接读端。
            shared_worker_lock: 全部 Worker 共用的进程锁，透传到子进程初始化路径。
                当前尚未接入实际的 Worker 侧共享资源使用逻辑。
            is_driver_worker: 当前 rank 是否承担 driver 职责，单机纯 TP 时仅 rank 0 为 True。
            inherited_fds: fork 模式下子进程应关闭的父端 FD 列表；spawn 时传 None。

        Returns:
            unready_worker_handle: 保存 proc、rank、ready_reader、death_writer 的
                父进程管理句柄。返回时只完成 proc.start()，还没有等到 READY。

        Side Effects:
            创建两条单向 Pipe 和 daemon 子进程；启动后父进程关闭自己不使用的
            ready_writer、death_reader。父进程必须继续持有 death_writer，
            才不会让子进程将写端关闭误认为父进程退出。
        """
        context = get_mp_context()  # 使用与 Executor 相同的进程启动配置创建 Pipe / Process。
        # Pipe(duplex=False) 依次返回接收端、发送端；READY 是子进程写给父进程的通知。
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # 死亡通知方向相反：父进程保留写端，子进程监控读端的 EOF，不靠周期性发送心跳。
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()  # 只扩充本次子进程的副本，不修改启动循环维护的列表。
            # 当前 Worker 自己也不应保留这两个父端 FD，尤其是死亡通知写端。
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        # 这些关键字参数交给子进程入口；管道只用于启动 / 生命周期，任务走 input_shm_handle。
        process_kwargs: dict[str, Any] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,  # 子进程用它发送 READY 状态与自己的结果队列句柄。
            "death_pipe": death_reader,  # 子进程用它观察父进程关闭写端。
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # 子进程入口应关闭 fork 意外继承的父端和其他 Worker 的管道 FD。
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # 创建 Worker 子进程对象；它应运行 Worker 的任务循环，Executor 留在父进程。
        proc = context.Process(
            target=WorkerProc.worker_main,  # 子进程入口负责初始化、READY、队列握手与 RPC 循环。
            kwargs=process_kwargs,  # spawn 时由 multiprocessing 序列化并交给新解释器。
            name=f"VllmWorker-{rank}",  # 日志和调试工具中用于区分不同 rank 的进程名。
            daemon=True,  # 将其作为受父进程管理的守护子进程。
        )

        proc.start()  # 发起子进程启动，不等待设备初始化、模型加载或队列连接完成。

        # 父进程只保留自己的管道端点，避免多余引用阻止对端收到 EOF。
        ready_writer.close()  # READY 的发送端归子进程持有，父进程只读取 ready_reader。
        death_reader.close()  # 死亡通知读端归子进程持有，父进程只保留 death_writer。
        # 保留 death_writer 至正常运行结束或启动失败清理，随后关闭它通知子进程退出。
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        """在父进程中连接 READY 消息携带的结果队列，并转换 Worker 管理句柄。

        由 wait_for_ready() 在确认 status == READY 后调用。当前子进程创建
        MessageQueue(1, 1)，所以父进程以该队列唯一读者 rank 0 的身份连接，
        这里的 0 与 proc_handle.rank 所表示的 Worker 编号不是同一个编号空间。

        Args:
            handles: 子进程的 READY 消息；handle 是其结果队列句柄，
                peer_response_handles 是上游保留的远端句柄列表，当前单机为空。
            proc_handle: 父进程持有的启动期管理句柄，保存子进程、rank 和管道。

        Returns:
            worker_proc_handle: 已附加结果队列读端的运行期管理句柄；进程和
                死亡通知写端沿用 proc_handle，后续仍需执行队列就绪握手。

        Side Effects:
            映射共享内存、连接结果队列 socket，但不创建新子进程、不读取实际
            模型结果，也不等待队列握手。远端连接分支沿用上游协议，单机不进入。
        """
        response_handle = handles["handle"]  # 当前 Worker 创建并通过 READY 上报的结果队列。
        worker_response_mq: MessageQueue | None = None  # 尚未连接本地结果读端。
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)  # Executor 是唯一读者。
        peer_response_handles = handles["peer_response_handles"]  # 当前单机场景是空列表。
        # 保留上游端点转换方式；远端读者用 -1，只有提供远端地址的句柄才需要连接。
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,  # 继续持有原来的 Process、rank 和死亡通知写端。
            worker_response_mq,  # 父进程随后从该读端收取当前 Worker 的 RPC 结果。
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        """等待所有子进程上报 READY，连接结果队列并按 rank 返回管理句柄。

        Args:
            unready_proc_handles: 已完成 start() 的子进程管理句柄，READY 到达顺序
                不必与启动顺序一致；当前单机 rank 连续且从 0 开始。

        Returns:
            ready_proc_handles: 按 Worker rank 排列的 WorkerProcHandle 列表，
                每项已连接结果队列读端，但尚未执行后续队列 wait_until_ready()。

        Raises:
            Exception: 子进程上报非 READY 状态，或在发出 READY 前关闭管道。
                真正的初始化异常栈应由子进程输出，父进程在此报告统一启动失败。
                消息格式无效或响应队列句柄连接失败时，相关异常直接传播。

        Side Effects:
            阻塞读取 READY 管道并关闭每个已处理的接收端。本方法未设置等待
            超时，仍存活却一直不上报的 Worker 会使等待持续；失败后的进程
            退出清理由 _init_executor() 的 finally 负责。
        """
        e = Exception(  # 启动协议失败时向父进程调用方提供的统一错误，不替代子进程原始栈。
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        # 由可等待的 READY 接收端反查所属 Worker，允许同时监听所有子进程。
        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        # 提前按 Worker 数预留位置；随后按 rank 填入，不能按 READY 到达顺序 append。
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        # 每成功读取一个 READY 就移除一条管道，直到全部 Worker 都已完成启动上报。
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())  # 返回当前可读或已关闭的管道集合。
            for pipe in ready:
                # 本方法只登记 Connection，确保返回对象可执行 recv() 和 close()。
                assert isinstance(pipe, Connection)
                try:
                    unready_proc_handle = pipes.pop(pipe)  # 取出这条通知所属的启动句柄，不再重复等待。
                    response: dict[str, Any] = pipe.recv()  # 读取 status、结果队列 Handle 等启动信息。
                    if response["status"] != "READY":
                        raise e  # 收到初始化失败状态，停止等待并交给外层回收已启动进程。

                    # 沿用上游将全局 rank 映射到本地列表的方式；当前单机时 idx 就是 rank。
                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    # 连接子进程上报的结果队列读端，再把启动句柄转换为运行期管理句柄。
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True  # READY 前管道断开时，用统一启动失败代替底层 EOF 栈。
                    raise e from None  # 原始初始化失败原因应查看对应子进程的日志。

                finally:
                    pipe.close()  # READY 管道只使用一次，后续 RPC 不再通过它传递。

        # 每个合法 rank 都已填入对应位置；cast 仅收窄类型，不会复制列表或再次校验内容。
        return cast(list[WorkerProcHandle], ready_proc_handles)
    def shutdown(self) -> None:
        """在 Worker 子进程中停止队列等待，清理执行资源并销毁分布式环境。

        通常由 worker_main() 的 finally 在任务循环退出后调用。队列 shutdown()
        负责标记关闭和唤醒等待，不等于此处已释放全部共享内存或 socket；底层
        资源还取决于对象引用释放和进程结束。当前输出线程为守护线程，没有
        单独的退出哨兵或 join 协议，随 Worker 进程结束退出。

        Side Effects:
            先标记队列关闭并唤醒阻塞的任务读取，再委托真实 Worker 清理 Runner，随后
            清空本对象的队列引用并销毁模型并行组、默认分布式环境。清理异常
            继续传播；本方法没有为每一步单独捕获异常或提供额外的重复关闭保护。
        """
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()  # 取消任务读端等待，避免继续阻塞接收新 RPC。
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()  # 标记结果队列关闭；写端没有本地读者取消通知。
        self.worker.shutdown()  # Wrapper 转发给真实 GPU Worker，清理模型、KV 与 Runner 资源。
        self.rpc_broadcast_mq = None  # 当前对象不再把任务队列当成可用端点。
        self.worker_response_mq = None  # 其他引用可能仍持有队列，不能据此认为底层资源立即释放。
        destroy_model_parallel()  # 先销毁 TP 等模型并行子组及其通信对象。
        destroy_distributed_environment()  # 再销毁 world / 默认进程组，结束当前进程的分布式状态。

    def monitor_death_pipe(
        self,
        death_pipe: Connection | None,
        shutdown_requested: threading.Event,
    ) -> None:
        """启动守护线程，通过死亡管道 EOF 观察 Executor 是否已经退出。

        Args:
            death_pipe: 子进程持有的接收端；父进程保持写端打开，退出或主动关闭
                写端后，本端 recv() 会收到 EOF。None 表示不启用这条监控路径。
            shutdown_requested: 当前子进程内各线程共享的退出标记，EOF 时置位，
                供 worker_main() 区分主动退出与未预期的运行异常。

        Side Effects:
            创建后台线程并阻塞等待管道关闭。观察到 EOF 时设置退出标记，标记
            两条队列关闭，唤醒任务读端，使任务线程离开阻塞的 dequeue() 并清理。
            不直接中断正在运行的 GPU 计算，也不在监控线程销毁 CUDA / 分布式资源。
        """
        if death_pipe is None:
            return  # 没有父进程存活管道时，不创建额外监控线程。

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue | None]) -> None:
            """等待父端断开，并取消启动时捕获的队列端点。

            Args:
                queues_to_shutdown: 当前任务、结果队列的引用快照，允许尚未建立
                    的端点为 None，不通过 self 再查找可能已被清空的成员。

            Side Effects:
                阻塞当前监控线程；EOF 后设置 shutdown_requested 并唤醒任务读取。
                其他管道异常只记录日志，不在本线程重新抛到任务主线程。
            """
            try:
                death_pipe.recv()  # 协议不需要心跳数据，写端关闭产生的 EOF 就是退出通知。
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()  # 主线程因队列取消抛异常时，可识别为已请求的关闭。
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()  # 标记关闭并取消本地读者等待；真实 Worker 清理由主线程 finally 执行。
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # 显式传入队列快照，避免通过绑定的 self 让监控线程持续持有整个 WorkerProc。
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),  # 两个方向的队列都参与关闭通知。
            daemon=True,  # 不让监控管道的线程单独阻止 Worker 进程退出。
            name="DeathPipeMonitor",  # 线程栈与调试工具中可据此识别父进程存活监控。
        ).start()

    @staticmethod
    def worker_main(*args: Any, **kwargs: Any) -> None:
        """作为子进程入口，组织 Worker 初始化、启动握手、任务循环和退出清理。

        make_worker_process() 在父进程把本方法设为 Process.target。当前调用
        全部使用关键字参数：先取出管道与继承 FD，剩余参数交给 WorkerProc
        构造函数。构造成功后先上报 READY，让父进程连接结果读端，再依次完成
        任务与结果队列的握手；交换顺序可能让父子进程相互等待。

        Args:
            *args: 透传给 WorkerProc 构造函数的位置参数，当前启动路径为空。
            **kwargs: WorkerProc 的构造参数，以及 ready_pipe、death_pipe 和
                inherited_fds。ready_pipe 是子进程发送 READY 的端点，death_pipe
                是观察父进程退出的接收端，inherited_fds 是 fork 后应关闭的多余 FD。

        Raises:
            SystemExit: SIGTERM / SIGINT 处理器请求退出时重新抛出，使进程结束。
            Exception: try 块外的启动参数提取，或 finally 中的清理自身失败时
                继续传播；try 块内的构造、握手和任务循环异常会记录后进入清理。

        Side Effects:
            安装信号处理器、关闭多余管道 FD，创建真实 Worker 及后台线程，
            通过 READY 管道上报结果队列句柄，并在当前线程持续执行 RPC。
            finally 关闭启动管理管道，再委托已构造成功的 WorkerProc 清理资源。
            SIGKILL 无法捕获，进程被它终止时不会执行本方法的 finally。
        """
        shutdown_requested = threading.Event()  # 子进程内共享退出状态，不是跨进程的 Event。

        def signal_handler(signum: int, frame: Any) -> None:
            """首次收到终止信号时请求退出，避免重复信号再次打断清理。

            Args:
                signum: 接收到的 SIGTERM 或 SIGINT 编号。
                frame: 信号处理器收到的当前 Python 栈帧，此处不使用。

            Raises:
                SystemExit: 尚未请求退出时置位标记，并让主线程离开当前执行路径。

            Side Effects:
                修改 shutdown_requested；已经置位时不重复抛出退出异常。
            """
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()  # 先标记，后续清理或重复信号可据此避免再次发起退出。
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()  # 信号在主线程处理，异常会进入本入口的退出及 finally 路径。

        signal.signal(signal.SIGTERM, signal_handler)  # 接收父进程 terminate() 发出的退出请求。
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C 也走同一套退出标记与清理流程。
        vllm_config = kwargs["vllm_config"]  # 完整启动配置仍留在 kwargs 中，随后透传给构造函数。

        worker: WorkerProc | None = None  # 只有构造完整成功后赋值，供 finally 判断能否调用实例清理。
        ready_writer: Connection | None = kwargs.pop("ready_pipe")  # 启动期 READY 发送端，握手结束后置 None。
        death_pipe: Connection | None = kwargs.pop("death_pipe", None)  # 父进程存活通知读端，不传给构造函数。
        # fork 会继承父进程和此前 Worker 的管道端点；多余写端会阻止对端观察到 EOF。
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)  # 只关闭父进程明确列出的多余 FD，保留当前子进程要用的两个端点。
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)
        try:
            rank = kwargs.get("rank", 0)  # 保留上游入口中的 Worker 编号别名，当前未另接 tracing / watchdog。

            worker = WorkerProc(*args, **kwargs)  # 完成设备、模型和队列端点初始化，尚未向父进程报告就绪。
            assert worker.worker_response_mq is not None  # 单机 Worker 必须能上报自己的结果队列句柄。

            worker.monitor_death_pipe(death_pipe, shutdown_requested)  # 尽早建立父进程退出时的唤醒路径。
            # READY 表示初始化完成且结果队列可供连接；不表示双方的队列握手已经完成。
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,  # 父进程 wait_for_ready() 用它判断启动成功。
                    "handle": worker.worker_response_mq.export_handle(),  # 父进程据此建立结果读端。
                    "peer_response_handles": worker.peer_response_handles,  # 上游协议保留项，当前单机为空。
                }
            )
            # 必须与 Executor 保持同一顺序：先任务广播，再当前 Worker 的结果队列。
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()  # 等待 Executor 已订阅结果队列，之后才允许正常回包。
            ready_writer.close()  # READY 管道任务完成，后续调用全部经消息队列传递。
            ready_writer = None  # 同时作为启动阶段结束标记，异常分支据此选择日志。

            worker.worker_busy_loop()  # 正常服务阶段：阻塞接收 RPC，并串行执行真实 Worker 方法。
        except Exception:
            # 普通方法异常由 _execute_worker_rpc() 按回包约定发送 FAILURE。
            # 到达这里的初始化、握手或队列异常只记录日志并退出，不额外构造失败响应。
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")  # READY 尚未完成或仍处于队列握手阶段。
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")  # 已进入服务阶段，且异常前未收到退出请求。

            # 已决定退出后不让后续 SIGTERM / SIGINT 再次抛出 SystemExit，减少清理被反复打断。
            shutdown_requested.set()
        except SystemExit as e:
            # SIGTERM / SIGINT 的处理器会主动抛出 SystemExit；SIGKILL 不会进入任何 Python 异常分支。
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            raise e  # 保留进程退出语义，finally 仍会执行；不能吞掉退出异常后继续提供服务。
        finally:
            if ready_writer is not None:
                ready_writer.close()  # 启动失败时关闭写端，让父进程等待 READY 的 recv() 能收到 EOF。
            if death_pipe is not None:
                death_pipe.close()  # 释放当前子进程持有的父进程存活通知端点。
            if worker is not None:
                worker.shutdown()  # 只清理已经完整构造返回的实例，覆盖正常退出和运行异常。

    class ResponseStatus(Enum):
        """结果消息的状态字段，接收端据此区分返回值与远端执行失败。"""

        SUCCESS = auto()  # payload 是方法返回值，None 也属于合法的成功结果。
        FAILURE = auto()  # payload 是错误字符串，原始 traceback 在子进程日志中记录。

    def enqueue_output(self, output: Any) -> None:
        """将输出整理为可传输的成功或失败消息，写入当前 Worker 的结果队列。

        Args:
            output: 普通方法返回值、AsyncModelRunnerOutput，或待回传的 Exception。
                异步模型输出须在本进程调用 get_output()，等待拷贝完成并整理为
                ModelRunnerOutput，再传给 Executor，不能直接发送其中的 CUDA 事件。

        Side Effects:
            可能等待设备到 CPU 的拷贝，随后发送 (ResponseStatus, payload)。
            get_output() 失败会记录异常并转换为 FAILURE；队列发送异常直接传播。
            若结果队列已置为 None，则跳过发送，不再尝试访问已关闭的端点。
        """
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()  # 在产生设备结果的 Worker 进程内等待并回收输出。
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e  # 复用下面的错误消息封装路径，让 Executor 收到失败而不是模型结果。
        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))  # 跨进程只发送错误文本。
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)  # None 等普通返回值也按成功协议发送。
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)  # 发往当前 Worker 专属结果队列，唯一读者是 Executor。

    def handle_output(self, output: Any) -> None:
        """按调度模式选择当前 RPC 结果的回收线程，并保持回包顺序。

        Args:
            output: 当前 RPC 的返回值或异常，可能包含尚未完成 D2H 的异步输出。

        Side Effects:
            异步调度时把所有结果交给同一个进程内 FIFO 队列，由输出线程统一
            处理；普通返回值也不能绕过前面的异步结果，否则共享结果队列中的
            响应顺序会与 FutureWrapper 的回收顺序错位。同步模式直接在任务
            主线程整理并发送结果，可能阻塞到设备输出可读。
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)  # 只在当前进程交接对象引用，不在这里序列化或等待 D2H。
        else:
            self.enqueue_output(output)  # 同步模式由任务主线程完成回收和发送。

    def async_output_busy_loop(self) -> None:
        """在 Worker 的输出守护线程中，按顺序回收设备结果并向 Executor 回包。

        Side Effects:
            先将本线程绑定到真实 Worker 的 CUDA 设备，再持续阻塞读取进程内
            输出队列并调用 enqueue_output()。本线程不执行模型 RPC；主线程
            可以继续接收后续任务。当前没有退出哨兵，依赖 Worker 进程结束
            终止守护线程；未捕获的线程异常也不会自动进入 worker_main() 的异常分支。
        """
        from nanovllm.platforms import current_platform
        if hasattr(self.worker, "device"):
            # 新线程不继承主线程的当前 CUDA device，必须显式设置，避免回收时误用 GPU 0。
            current_platform.set_device(self.worker.device)
        while True:
            output = self.async_output_queue.get()  # FIFO 顺序与主线程交付 RPC 结果的顺序一致。
            self.enqueue_output(output)  # 只有这一步才等待异步拷贝并通过跨进程消息队列发送。

    def worker_busy_loop(self) -> None:
        """在 Worker 主线程中串行接收任务广播，并执行真实 Worker 方法。

        Side Effects:
            持续从任务读端消费 RPC，空闲时允许无限等待；队列取消或通信异常
            向 worker_main() 传播，由它判断是否属于退出请求并执行清理。
            模型输出可交给辅助线程回收，但同一个 Worker 的 RPC 执行仍是串行的。
        """
        # 调用顺序约定 READY 与握手已经完成；此断言只检查任务读端确实存在。
        assert self.rpc_broadcast_mq is not None
        while True:
            rpc_request = self.rpc_broadcast_mq.dequeue(indefinite=True)  # 空闲等待不作为长时间无进展告警。
            self._execute_worker_rpc(rpc_request)  # 在单独函数帧内执行，调用结束后释放方法参数等局部引用。
            del rpc_request  # 避免等下一条任务时仍持有大输入或共享 Tensor 的存储引用。

    def _execute_worker_rpc(
        self,
        rpc_request: tuple[str | bytes, tuple[Any, ...], dict[str, Any], int | None],
    ) -> None:
        """解析并执行一条 RPC，按约定选择是否回传结果或错误。

        Args:
            rpc_request: 四元组 (method, args, kwargs, output_rank)。method 是
                Worker 方法名，或 cloudpickle 编码的 callable；args / kwargs
                是调用参数。output_rank 为 None 表示全部 rank 回包，否则仅
                指定的全局 rank 回包，全部 rank 仍会执行方法本身。

        Side Effects:
            在当前主线程调用 WorkerWrapperBase 上的方法或绑定 callable，
            可能执行 GPU 计算、TP collective 或更新请求与 KV 状态。方法异常
            在子进程打印 traceback，按回包 rank 的约定交给 handle_output()。
            非回包 rank 的方法异常只记录日志，不通过其他 rank 的结果队列转发。
        """
        method, args, kwargs, output_rank = rpc_request  # 方法、调用参数以及回包选择共同构成调用协议。
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)  # Wrapper 的显式接口或其转发的真实 Worker 绑定方法。
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)  # callable 以执行端点为首个参数。
            output = func(*args, **kwargs)  # 各 rank 都必须执行，才能共同参与模型分片计算与通信。
            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)  # 回包筛选发生在执行之后，不会跳过其他 rank 的 TP 工作。
        except Exception as e:
            # Python 3.11 起可附加 traceback 备注；当前 Python 3.10 通过 logger.exception 输出完整栈。
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception("WorkerProc hit an exception.")
            # 错误同样遵守回包 rank 约定，enqueue_output() 会把异常转成 FAILURE 和错误字符串。
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        """按当前 TP 身份设置操作系统进程标题和后续日志的进程前缀。

        在设备初始化前后各调用一次：组未建立时使用 Worker，建立后 TP size
        大于 1 时追加 _TP{rank}，使模型加载和正式执行阶段的日志能区分分片。

        Args:
            enable_ep: 保留上游调用签名，当前实现不处理专家并行；调用方传 False。

        Side Effects:
            修改当前进程在系统工具中的标题，以及项目 formatter 使用的日志
            显示名；不创建通信组，也不改变已分配的 Worker rank 或 GPU 设备。
        """
        if not model_parallel_is_initialized():
            set_process_title(name="Worker")  # 当前还不能查询 TP group，先标记为普通 Worker。
            decorate_logs("Worker")  # 日志身份与操作系统进程标题通过两个接口分别设置。
            return

        tp_size = get_tp_group().world_size  # 当前 Worker 所在 TP group 的 rank 总数。
        tp_rank = get_tp_group().rank_in_group  # TP group 内的编号，不额外依赖节点编号。
        process_name = "Worker"  # TP=1 时保留短名称，不追加没有区分作用的 _TP0。
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        set_process_title(name=process_name)  # 供 ps / htop 等系统工具显示。
        decorate_logs(process_name)  # 供项目日志 formatter 显示，与现有 logger 对象继续配合使用。

def set_multiprocessing_worker_envs(local_world_size: int = 1) -> None:
    """在创建子进程前配置启动方式与 Worker 启动期的 OpenMP 线程预算。

    Args:
        local_world_size: 本机即将启动的 Worker 数，用于平分可用 CPU 线程预算。
            当前仅有一个模型副本，数值等于 TP size。

    Side Effects:
        根据 CUDA / WSL 状态可能将启动方式改为 spawn；GPU 后端还会写入
        OMP_NUM_THREADS 及项目设置标记，并修改当前进程的 Torch 线程数。
        后续子进程通过继承环境读取启动期线程预算。本方法不创建子进程。
    """
    _maybe_force_spawn()  # 已初始化 CUDA 或处于 WSL 时，避免使用不安全的 fork 路径。
    # 当前只支持 CUDA，所有 Worker 都使用启动期 OpenMP 线程预算。
    num_threads = startup_omp_num_threads(local_world_size)  # 每个 Worker 的启动线程数，至少为 1。
    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # 写入启动环境，让后续 Worker 使用同一预算。
    os.environ[OMP_NUM_THREADS_SET_BY_VLLM] = "1"  # 标记为项目设置，稳态 helper 可据此缩减线程。
    torch.set_num_threads(num_threads)  # 同时调整父进程；子进程的线程设置仍需其自身初始化。
    logger.debug(
        "Set OMP_NUM_THREADS=%d for %d worker process(es).",
        num_threads,
        local_world_size,
    )
