"""定义微批在请求轴与 packed token 轴上的切片，当前执行路径保留接口。"""
from typing import TypeAlias
from dataclasses import dataclass

@dataclass
class UBatchSlice:
    """关联同一微批的请求区间和 packed token 区间。

    两者都使用右开区间，调用方应提供明确的 start/stop 和连续、步长为 1 的切片。
    结构只描述边界，不复制 tensor，也不自动划分执行批次。
    """
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        """任一切片的起止位置相等时返回 True，不检查负长度或隐式切片边界。

        Returns:
            is_empty: 请求切片或 token 切片为空时为 True，不额外校验反向范围。
        """
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        """返回 token_slice 的 stop 与 start 之差，单位为 packed 输入 token。

        Returns:
            num_tokens: token_slice.stop - token_slice.start，单位为 packed 输入 token。
        """
        return self.token_slice.stop - self.token_slice.start
    
UBatchSlices: TypeAlias = list[UBatchSlice]