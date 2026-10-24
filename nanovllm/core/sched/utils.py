import contextlib

from nanovllm.request import Request, RequestStatus


def check_stop(request: Request, max_model_len: int) -> bool:
    """追加真实 token 后检查停止条件，最小输出长度不能绕过上下文上限。

    Args:
        request: 已追加本轮 token、至少含一个真实输出的请求。
        max_model_len: prompt 与输出合计的硬长度上限。

    Returns:
        stopped: 是否已经停止；未满足 min_tokens 且未到上下文上限时返回 False。

    Side Effects:
        命中 EOS、停止 token 或长度边界时更新 request.status；用户指定的
        停止 token 还写入 stop_reason。EOS 是否启用取决于 Engine 注入的 EOS 值。
    """
    sampling_params = request.sampling_params
    assert sampling_params is not None
    # 保留上游 EOS、stop token、长度限制的判断顺序；硬上限不能被 min_tokens 屏蔽。
    if (
        request.num_output_tokens < sampling_params.min_tokens
        and request.num_tokens < max_model_len
    ):
        return False
    # 只检查刚追加的真实输出，不读取异步占位或 GPU tensor。
    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """删除目标元素，保留剩余列表的相对顺序。

    Args:
        lst: 待过滤的原列表。
        items_to_remove: 待删除集合，不存在的目标忽略。

    Returns:
        remaining_items: 空集合返回原列表；单目标原地删除首次匹配项；多目标返回过滤后的新列表。
            调用方必须使用返回值，不能假设所有分支都原地修改。

    Side Effects:
        单目标分支修改 lst；当前请求队列成员唯一，只删除首次匹配项即可。
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # 常见的单请求结束路径不重建整个列表，保持现有列表引用。
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # 多请求同时结束时一次过滤，避免重复进行线性删除。
    return [item for item in lst if item not in items_to_remove]
