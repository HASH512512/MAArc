from __future__ import annotations

import numpy as np


class CoordConv:
    trans_mat: np.ndarray

    def __init__(
        self,
        dl: tuple[float, float],
        ul: tuple[float, float],
        ur: tuple[float, float],
        dr: tuple[float, float],
    ):
        x0, y0 = dl
        x1, y1 = ul
        x2, y2 = ur
        x3, y3 = dr
        a, b, c = x3 - x2, x1 - x2, x1 + x3 - x0 - x2
        d, e, f = y3 - y2, y1 - y2, y1 + y3 - y0 - y2

        g = b * d - a * e
        h = (a * f - c * d) / g
        g = (c * e - b * f) / g

        c, f = x0, y0
        a = (g + 1) * x3 - c
        b = (h + 1) * x1 - c
        d = (g + 1) * y3 - f
        e = (h + 1) * y1 - f

        self.trans_mat = np.array(((a, b, c), (d, e, f), (g, h, 1))).T

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        x_, y_, z_ = np.array((x, y, 1)) @ self.trans_mat
        return x_ / z_, y_ / z_
