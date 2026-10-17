"""控制启动完成后的 Python 循环 GC 扫描范围，不管理 CUDA 显存或 Graph pool。"""

import gc


def freeze_gc_heap() -> None:
    """预热完成后将当前 GC 跟踪对象移入永久代，减少推理阶段的循环 GC 扫描。

    调用时模型、正式 KV cache 和 CUDA Graph 等常驻对象已创建。沿用 vLLM
    逐代收集再 freeze 的顺序，使启动期间产生的临时循环垃圾先得到清理。

    Side Effects:
        改变当前进程的 Python GC 状态，退出函数后不会自动恢复。freeze 不会
        禁用新对象的 GC，也不阻止引用计数归零时释放对象；它只让冻结对象
        不再参与后续循环 GC 扫描。需要恢复该扫描范围时应调用 gc.unfreeze()。
        此接口不会冻结 GPU tensor 地址、捕获 CUDA Graph 或锁定 allocator。
    """
    # 先收集第 0 代短生命周期对象，存活对象晋升到更老的一代。
    gc.collect(0)
    # 再收集中间代，使启动期间存活的对象继续晋升。
    gc.collect(1)
    # 最后执行包含最老代的完整收集，清理尚可回收的循环引用。
    gc.collect(2)
    # 将当前仍被 GC 跟踪的对象移入永久代，后续自动循环 GC 跳过这些对象。
    gc.freeze()
