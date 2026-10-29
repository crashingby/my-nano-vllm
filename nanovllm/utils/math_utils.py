"""提供按整数倍向上取整的尺寸运算，调用方保证除数为正整数。"""

def cdiv(a: int, b: int) -> int:
    """返回 a/b 向上取整的整数商，用于计算覆盖全部元素所需的块数。

    Returns:
        num_blocks: a / b 向上取整的整数商，常用于计算所需块数。
    """
    return -(a // -b)

def round_up(x: int, y: int) -> int:
    """返回不小于 x 的最小 y 倍数，用于容量或维度对齐。

    Returns:
        rounded_size: 不小于 x 的最小 y 倍数，用于容量或维度对齐。
    """
    return ((x + y - 1) // y) * y

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator