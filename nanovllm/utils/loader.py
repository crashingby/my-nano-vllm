"""早期 safetensors 加载辅助接口；主执行链使用 model_executor.model_loader。"""
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """将完整 checkpoint tensor 原地复制到目标参数，不做打包或 TP 切分。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """遍历本地 safetensors 文件，按参数加载器把 checkpoint 写入模型。

    Args:
        model: 已构造参数的模型，可定义 packed_modules_mapping 和参数 weight_loader。
        path: 含一个或多个 safetensors 文件的本地目录，不下载远端权重。

    Side Effects:
        逐个文件打开 CPU checkpoint 并修改模型参数。打包映射传入 shard_id，
        普通参数使用自身加载器或默认 copy_；没有缺失权重的完整性检查。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})  # 原名到打包名/段号。
    # 逐文件释放 checkpoint 句柄，不把全部权重预先保存在内存列表中。
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 命中打包映射后只加载对应段，避免再次走普通参数路径。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for-else 只在没有打包匹配时执行，普通权重可复用默认复制器。
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
