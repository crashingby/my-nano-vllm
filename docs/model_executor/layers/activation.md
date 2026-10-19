# SiluAndMul：SiLU 门控激活

[`SiluAndMul`](../../../nanovllm/model_executor/layers/activation.py) 是模型前馈层中的激活模块，完成 **SiLU 激活与逐元素乘法**。它对应 SwiGLU 中对 gate 分支激活、再与 up 分支相乘的部分。

## 计算原理

输入最后一维按 `[gate, up]` 拼接，形状为 `[..., 2H]`。沿最后一维等分后，两个分支的形状均为 `[..., H]`：

$$
y = \operatorname{SiLU}(gate) \odot up
  = \frac{gate}{1 + e^{-gate}} \odot up
$$

- **gate 分支**先经过 SiLU，生成对特征进行调制的门控激活值。
- **up 分支**与激活后的 gate 在相同位置相乘，$\odot$ 表示逐元素乘法。
- **输出形状**为 `[..., H]`，前导维度保持不变，最后一维减半；输出保持输入的设备和精度，输入内容保持原值。

这里的 $H$ 表示传入本层的单个分支宽度。计算在每个元素位置独立进行，各 token 之间没有数据依赖。

## 在前馈层中的位置

以 [`Qwen3MLP`](../../../nanovllm/model_executor/models/qwen3.py) 为例，gate 和 up 先通过合并投影一起生成，再经过本层，最后由 down 投影映射回模型的隐藏维度：

```python
gate_up = self.gate_up_proj(x)  # 一次投影生成拼接的 gate 和 up。
x = self.silu_and_mul(gate_up)  # SiluAndMul：SiLU(gate) * up。
x = self.down_proj(x)         # 将中间特征映射回模型隐藏维度。
```

`SiluAndMul` 本身没有可学习参数，投影权重由前后的线性层持有。

## 层接口与算子入口

模型通过 `SiluAndMul()` 创建模块，通过 `layer(x)` 执行计算。构造时可以显式指定 `backend`，否则读取当前配置的 `kernel_config.silu_and_mul_backend`；没有配置上下文时默认使用 `torch`。

**后端在构造时绑定**，后续 `forward()` 直接调用已选实现。三个后端共享 [`silu_and_mul` 算子](../../../nanovllm/ir/ops/activation.py)的计算语义：

| 后端 | 层绑定的实现 |
| --- | --- |
| `torch` | 算子的 `native` 实现，使用共享的 `torch.compile` 包装。 |
| `triton` | 算子注册的 [Triton 实现](../../../nanovllm/kernels/triton/activation.py)。 |
| `cutedsl` | 算子注册的 [CuTe DSL 实现](../../../nanovllm/kernels/cute/activation.py)。 |

后端实现特点与性能对比见 [silu_and_mul 算子文档](../../ir/ops/activation.md)；线程映射、访存方式、持久化调度和自动调优的具体代码可沿表中的实现入口继续阅读。
