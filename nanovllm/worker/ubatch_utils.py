from typing import TypeAlias
from dataclasses import dataclass

@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start
    
UBatchSlices: TypeAlias = list[UBatchSlice]