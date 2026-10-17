"""基于完整 NanovllmConfig 的同步文本生成入口。"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """使用 ``LLM(vllm_config)`` 创建支持 generate() 的同步 Engine。

    当前直接继承 LLMEngine 的构造和请求处理接口；调用方先组装 NanovllmConfig，
    模型路径、调度容量、KV 策略和编译选项均从该对象传入执行链。
    """
