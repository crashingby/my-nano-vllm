// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/types.h>

#include <stdexcept>
#include <vector>

/**
 * @brief 创建共享原始 CUDA 数据地址、但不拥有原始分配的 Tensor。
 * @param tensor 输入 CUDA Tensor；shape、stride、dtype 和 device 原样保留。
 * @return 不复制数据的非 owning Tensor；不维持输入 Tensor 或 storage 的生命周期。
 *
 * 调用方必须保证底层地址仍然有效。CUDA Graph 路径由存活的 Graph pool
 * 保留捕获期间的分配；共享 pool 的其他图可能覆盖这段数据。
 */
inline torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // 与 vLLM 一致，只接受 CUDA；这个接口本身不启动 GPU kernel。
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  void* data_ptr = tensor.data_ptr();  // 从当前视图的起始位置读取地址，包含原 storage offset。
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();  // 保留非连续视图的布局。
  auto options = tensor.options();  // 保留 dtype 和具体 CUDA device。

  // 不传 deleter，也不捕获原 Tensor；新 storage 不负责释放或保活原始分配。
  return torch::from_blob(data_ptr, sizes, strides, options);
}
