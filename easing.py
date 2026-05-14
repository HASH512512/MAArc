from enum import Enum
from functools import partial
from math import pi, sin, cos
from typing import Callable


EasingFunction = Callable[
    [tuple[float, float, float], tuple[float, float, float], float],
    tuple[float, float, float],
]


class _EasingValue:
    func: EasingFunction

    def __init__(self, func: EasingFunction) -> None:
        self.func = func

    def __call__(
        self, start: tuple[float, float, float], end: tuple[float, float, float], t: float
    ) -> tuple[float, float, float]:
        return self.func(start, end, t)


def _easing_linear(
    start: tuple[float, float, float], end: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    x0, y0, z0 = start
    x1, y1, z1 = end
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t


def _easing_cubic_bezier(
    start: tuple[float, float, float], end: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    start_scale = (1 - t) * (1 + 2 * t)
    end_scale = t * (3 - 2 * t)
    curve_start = (start[0] * start_scale, start[1], start[2] * start_scale)
    curve_end = (end[0] * end_scale, end[1], end[2] * end_scale)
    return _easing_linear(curve_start, curve_end, t)


def _easing_sinus(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
    x: str,
    z: str | None = None,
) -> tuple[float, float, float]:
    x0, y0, z0 = start
    x1, y1, z1 = end
    if x == "si":
        sx = sin(t * pi / 2)
    elif x == "so":
        sx = 1 - cos(t * pi / 2)
    else:
        raise RuntimeError(f"unknown easing type x = {x}")
    if z == "si":
        sz = sin(t * pi / 2)
    elif z == "so":
        sz = 1 - cos(t * pi / 2)
    else:
        sz = t
    return x0 + (x1 - x0) * sx, y0 + (y1 - y0) * t, z0 + (z1 - z0) * sz


class Easing(Enum):
    Linear = _EasingValue(_easing_linear)
    CubicBezier = _EasingValue(_easing_cubic_bezier)
    Si = _EasingValue(partial(_easing_sinus, x="si"))
    SiSi = _EasingValue(partial(_easing_sinus, x="si", z="si"))
    SiSo = _EasingValue(partial(_easing_sinus, x="si", z="so"))
    So = _EasingValue(partial(_easing_sinus, x="so"))
    SoSo = _EasingValue(partial(_easing_sinus, x="so", z="so"))
    SoSi = _EasingValue(partial(_easing_sinus, x="so", z="si"))


if __name__ == "__main__":
    print(_easing_linear((0, 1, 0), (1, 1, 0), 0.2))
    print(_easing_cubic_bezier((0, 0, 0), (1, 1, 0), 0.2))
    print(Easing.So.value((0, 1, 0), (1, 1, 0), 0.2))
    print(Easing.CubicBezier)
    print(Easing.SiSi)
