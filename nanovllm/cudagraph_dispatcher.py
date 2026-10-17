"""根据 packed token 数和 batch 类型选择 CUDA Graph 运行模式及形状 key。

沿用 vLLM V1 的尺寸映射、key 初始化、FULL 优先匹配和捕获顺序。当前只支持
单 token decode，省略 LoRA、spec decode 和尚未实现的 compile_sizes 校验。
保留 PIECEWISE 的 key 规则不代表编译链已接通；本模块不执行 GPU 工作。
"""

from collections.abc import Set as AbstractSet
from dataclasses import replace

from nanovllm.config import CUDAGraphMode, NanovllmConfig
from nanovllm.forward_context import BatchDescriptor


class CudagraphDispatcher:
    """维护合法捕获 key，为本轮执行选择具体模式和 padding 后的描述符。

    Runner 在 Attention 能力解析后初始化 keys；启动捕获流程读取
    get_capture_descs()，在线执行调用 dispatch()。key 存在只表示允许使用
    该形状，实际图对象和捕获完成状态由后续 Graph Wrapper 管理。

    Attributes:
        vllm_config: 保留上游名称，引用当前 nano-vLLM 总配置。
        compilation_config: 已解析尺寸的配置引用，初始化后不应再修改。
        uniform_decode_query_len: 每个有效 decode 请求的 query 数，当前为 1。
        cudagraph_keys: PIECEWISE/FULL 分别对应一组不可变、可哈希的描述符。
        keys_initialized: 初始化是否完成；NONE 也属于已初始化状态。
        cudagraph_mode: 经后端能力解析的策略，初始化前为 NONE。
        _bs_to_padded_graph_size: token 数到最小兼容捕获尺寸的查表数组。
    """

    def __init__(self, vllm_config: NanovllmConfig) -> None:
        """保存配置并创建空 key 集合，不探测 GPU 或创建 CUDA 资源。

        Args:
            vllm_config: 总配置，提供规范化捕获尺寸和 Scheduler 请求容量。
        """
        self.vllm_config = vllm_config  # 保留引用，读取 Scheduler 请求槽位上限。
        self.compilation_config = vllm_config.compilation_config  # 已解析的尺寸配置。
        self.uniform_decode_query_len = 1  # 当前没有 spec decode 的多 query 情况。
        # NONE 不需要图；组合配置策略不能直接作为 Wrapper 的运行模式。
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        self.keys_initialized = False  # 未初始化时 dispatch 回退普通执行。
        self.cudagraph_mode = CUDAGraphMode.NONE  # 初始化后才使用最终配置策略。

    def _compute_bs_to_padded_graph_size(self) -> None:
        """预计算有效 token 数到最小兼容捕获尺寸的映射。

        沿上游 bs 命名，其单位是 packed token，不是一般 batch 的请求数。
        捕获尺寸必须已升序排列，且末项等于配置上限。例如 [1, 2, 4, 8]
        生成 [0, 1, 2, 4, 4, 8, 8, 8, 8]；0 不因此成为捕获 key。

        Side Effects:
            创建 _bs_to_padded_graph_size，供在线分派常数时间查表。

        Raises:
            AssertionError: Graph 启用后仍未解析捕获尺寸或上限。
        """
        max_size = self.compilation_config.max_cudagraph_capture_size  # 查表上界。
        capture_sizes = self.compilation_config.cudagraph_capture_sizes  # 升序分档。
        # None 表示遗漏总配置解析；空列表/零上限则可以表示过滤后的空计划。
        assert max_size is not None, (
            "Maximum cudagraph capture size must be set when cudagraphs are enabled."
        )
        assert capture_sizes is not None, (
            "Cudagraph capture sizes must be set when cudagraphs are enabled."
        )
        self._bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        # 相邻分档构成 [start, end)；末尾哨兵使最大尺寸本身也被写入。
        for end, start in zip(
            capture_sizes + [max_size + 1],
            [0] + capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self._bs_to_padded_graph_size[bs] = start  # 分档点无需 padding。
                else:
                    self._bs_to_padded_graph_size[bs] = end  # 中间值向上补齐。
        # 上游还校验 compile_sizes 不被 padding 改变；当前没有该配置和编译链。

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
    ) -> BatchDescriptor:
        """从有效 token 数构造执行形状，不修改输入或 Attention metadata。

        Args:
            num_tokens: 未 padding 的非负 token 数，调用前已确认在查表范围内。
            uniform_decode: 是否按 uniform_decode_query_len 构造 decode 形状；
                策略不包含 FULL 时会归一为 False。

        Returns:
            padded token 数、请求槽位容量和归一化 uniform 标记的描述符。
            num_reqs 不是从本轮真实请求列表中统计的数量。
        """
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs  # 请求容量。
        uniform_decode_query_len = self.uniform_decode_query_len  # 当前固定为 1。
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]  # 最小兼容分档。

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            # decode 槽位由 padded token 数推导，再受请求容量限制。
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False  # mixed key 不按真实请求的 query 长度区分。
            num_reqs = min(num_tokens_padded, max_num_seqs)  # 请求槽位不超配置容量。

        return BatchDescriptor(
            num_tokens=num_tokens_padded,  # key 保存执行尺寸，不保存本轮有效行数。
            num_reqs=num_reqs,  # FULL 精确匹配，PIECEWISE 稍后放宽为 None。
            uniform=uniform_decode,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ) -> None:
        """登记一个合法 key，相同描述符自动去重，不创建图对象。

        Args:
            runtime_mode: 具体捕获模式 PIECEWISE/FULL，不能传组合策略或 NONE。
            batch_descriptor: 与捕获及运行时分派一致的执行形状。

        Side Effects:
            修改 cudagraph_keys；不设置 keys_initialized。
        """
        # 集合按 Wrapper 直接消费的运行模式划分，而不是按组合策略划分。
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ) -> None:
        """在 Attention 能力解析之后建立尺寸映射和合法捕获 key。

        Args:
            cudagraph_mode: 已解析的最终策略；本类不查询后端能力，也不降级。
            uniform_decode_query_len: 保留上游参数；当前普通 decode 只支持 1。

        Side Effects:
            写入策略、尺寸查表和 key 集合，最后设置 keys_initialized。
            同上游按启动期一次初始化使用；更改策略应创建新的 Dispatcher，
            避免旧 key 和已捕获图混用。

        Raises:
            ValueError: 请求当前未支持的多 query decode。
            AssertionError: 启用 Graph 时尺寸尚未完成配置解析。
        """
        # 禁止按不支持的长度筛 key，却仍按成员中的长度 1 构造描述符。
        if uniform_decode_query_len != 1:
            raise ValueError("Only uniform_decode_query_len=1 is supported")
        self.cudagraph_mode = cudagraph_mode  # 后端能力解析由调用方提前完成。

        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True  # 禁用也是已确定结果，无需创建尺寸查表。
            return

        self._compute_bs_to_padded_graph_size()  # 先建立 token 分档映射。

        # mixed 分支处理纯 FULL、纯 PIECEWISE 及组合策略的非 decode 部分。
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when mixed mode is enabled."
            )
            # 当前没有 LoRA 维度，每个尺寸只生成一个 mixed key。
            for bs in self.compilation_config.cudagraph_capture_sizes:
                batch_desc = self._create_padded_batch_descriptor(bs, False)
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    # 分段图按 token 数区分，请求数交由图外 Attention 处理。
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # 组合策略将 decode 单独设为 FULL 时，再生成 uniform 专用 key。
        # 纯 FULL 已由 mixed 分支覆盖，不额外添加另一组 uniform key。
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )  # 普通 decode 的可用尺寸还受请求容量限制。
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]  # 超过 decode 请求容量的候选尺寸不生成 FULL key。
            for bs in cudagraph_capture_sizes_for_decode:
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(bs, True),
                )

        self.keys_initialized = True  # 所有集合就绪后才开放在线 key 匹配。

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """为本轮 packed batch 选择具体执行模式，以及与该模式匹配的形状 key。

        Runner 在识别本轮 batch 类型之后、准备 padding 输入之前调用本接口。
        输入描述真实调度的有效 token 数，输出描述模型将采用的执行形状。
        例如三个单 token decode 请求可以匹配四槽位 FULL key；这一步只返回
        计划，实际补齐 input_ids、positions 和 Attention metadata 由 Runner 完成。

        分派过程：
            1. 根据 valid_modes 和 invalid_modes 得到本轮允许的具体模式。
            2. 未初始化、禁用、未解析上限、超过捕获范围或只允许 NONE 时，
               直接返回普通执行描述符，不访问 padding 查表。
            3. 将有效 token 数向上补到最小兼容捕获尺寸，并推导请求槽位容量。
               只有组合策略单独保留 uniform 标记；纯 FULL 使用一般 batch key。
            4. 优先查询 FULL 集合，要求 token 数、请求槽位数和 uniform 全部匹配。
               未命中或本轮禁止 FULL 时，再把请求维度放宽为 None、uniform
               设为 False，查询 PIECEWISE 集合。
            5. 两种集合都未命中时，允许 NONE 则按原有效 token 数普通执行；
               禁止 NONE 则断言失败，不能返回一个未登记的捕获 key。

        本接口不检查某张图是否已经捕获。命中 key 表示该形状属于合法分派集合，
        实际图条目及回放由 Wrapper 管理；在线执行前应已完成所需图的启动捕获。

        Args:
            num_tokens: Scheduler 本轮实际调度的 packed token 总数，尚未 padding；
                调用方保证为非负整数。一个请求可以贡献多个 query token，
                所以这个值一般不等于真实请求数；普通单 token decode 时才相等。
            uniform_decode: Runner 是否确认所有有效请求都处于 uniform decode，
                每请求 query 长度等于 uniform_decode_query_len，当前为 1。
                True 不要求历史 KV 长度相同；False 不会触发自动识别。
                本类不读取 query_start_loc，错误标记可能选择不兼容的 FULL key。
            valid_modes: 本轮允许尝试的具体模式 NONE、PIECEWISE、FULL 的集合，
                不能用 FULL_DECODE_ONLY 等组合策略表达一次执行。
                None 或空集合沿上游语义使用默认全集；{NONE} 才表示明确普通执行。
                这是匹配限制，不改变 FULL 优先于 PIECEWISE 的选择顺序。
            invalid_modes: 从允许集合中排除的具体模式。例如本轮 Attention
                条件不能完整捕获时传 {FULL}，允许尝试已有 PIECEWISE key 或 NONE。
                None/空集合不排除模式；过滤不修改调用方的集合或已登记的 key。

        Returns:
            二元组 (runtime_mode, batch_descriptor)：
            - FULL：描述符包含 padded token 数、请求槽位容量和归一化 uniform
              标记，可作为完整图的精确 key；num_reqs 包含 padding 槽位，
              不表示本轮真实请求数。
            - PIECEWISE：保留 padded token 数，num_reqs=None、uniform=False，
              表示分段计算不按 Attention 请求结构区分 key，不表示零个请求。
            - NONE：返回 BatchDescriptor(num_tokens)，保留未 padding 的有效
              token 数，num_reqs=None、uniform=False，不要求准备 Graph padding。

            Runner 应保留原始有效 token/请求数量，用于选择有效 logits 行及回写
            请求状态；不能把返回的 padded 执行容量当成真实调度数量。

        Raises:
            AssertionError: 模式过滤后集合为空；或已经进入 key 匹配阶段却没有
                命中任何图，并且调用方禁止 NONE 回退。

        Notes:
            早返回沿用上游行为：未初始化、禁用、上限未解析或超范围时直接返回
            NONE，即使 valid_modes 没有包含 NONE。因此模式集合不能作为覆盖
            所有状态的严格执行保证，需要禁止普通执行的调用方还应检查返回值。
            当前 PIECEWISE 编译链未接通；保留这种分派规则不等于能实际分段回放。

        Examples:
            捕获尺寸为 [1, 2, 4, 8]、请求容量至少为 8，策略为 FULL_DECODE_ONLY：
            dispatch(3, True) 返回 (FULL, BatchDescriptor(4, 4, True))；
            dispatch(3, False) 不命中 decode 专用 key，返回 (NONE, BatchDescriptor(3))。
            同样尺寸在 FULL_AND_PIECEWISE 分派规则下，dispatch(3, False) 可以
            返回 (PIECEWISE, BatchDescriptor(4, None, False))；单独排除 FULL 时，
            即使本轮为 uniform decode，也优先尝试这个放宽后的 PIECEWISE key。

        Side Effects:
            只读取配置、尺寸查表和合法 key 集合，创建本次返回的描述符。
            不修改 Dispatcher 状态、请求、输入 tensor 或调用方集合，
            不执行 tensor padding、CUDA 分配、捕获或回放。
        """
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()
        if invalid_modes:
            # 新建差集，避免 -= 修改调用方传入的可变 set。
            allowed_modes = allowed_modes - invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )
        max_size = self.compilation_config.max_cudagraph_capture_size  # 越界不查表。

        # 这些状态直接普通执行；NONE 描述符不应携带 padded token 数。
        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or max_size is None
            or num_tokens > max_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        # 只有组合策略单独区分 uniform；纯 FULL 使用更一般的 mixed key。
        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(num_tokens, normalized_uniform)

        if CUDAGraphMode.FULL in allowed_modes:
            batch_desc_to_check = batch_desc  # FULL 保留请求槽位数和 uniform 标记。
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc_to_check

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            # FULL 不可用时，尝试只按 token 尺寸区分的分段 key。
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        # 匹配失败必须允许普通执行，不能虚构一个尚未登记的 Graph key。
        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """返回捕获计划：PIECEWISE 在前、FULL 在后，各模式尺寸降序。

        Returns:
            (具体模式, 描述符列表) 的列表，未初始化、禁用或无 key 时为空。
            decode 捕获尺寸 [1, 2, 4] 返回 FULL 下的 4、2、1 三个描述符。

        Side Effects:
            创建排序用列表，不修改 key 集合，也不创建或检查实际图对象。
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result: list[tuple[CUDAGraphMode, list[BatchDescriptor]]] = []
        # 相同 token 尺寸在组合策略下可能分别拥有两种模式的 key。
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            descs = list(self.cudagraph_keys[mode])  # 复制 set 后排序，不改变原集合。
            if descs:
                # 大图先捕获，便于后续小图在捕获流程中复用图内存池。
                descs.sort(key=lambda d: d.num_tokens, reverse=True)
                result.append((mode, descs))

        return result
