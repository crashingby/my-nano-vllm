"""导出逻辑算子注册入口；具体算子和 provider 由后续模块显式导入登记。"""

from .op import IrOp, IrOpImpl, register_op

__all__ = ["IrOp", "IrOpImpl", "register_op"]
