// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/library.h>

#include "ops.h"

// 使用独立命名空间，避免与同进程导入的 vLLM _C 算子重名。
TORCH_LIBRARY(_nanovllm_C, ops) {
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);
}
