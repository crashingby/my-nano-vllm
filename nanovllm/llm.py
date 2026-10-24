"""基于完整 NanovllmConfig 的阻塞式文本生成入口，支持同步或异步调度。"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """使用 ``LLM(vllm_config)`` 创建支持阻塞式 generate() 的 Engine。

    当前直接继承 LLMEngine 的构造和请求处理接口；调用方先组装 NanovllmConfig，
    模型路径、调度容量、KV 策略和编译选项均从该对象传入执行链。
    手动驱动请求时调用 step_fn()，由初始化配置选择同步或队列执行路径。
    """
