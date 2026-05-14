from __future__ import annotations

import numpy as np


def _homography_from_points(
    source_points: list[tuple[float, float]],
    target_points: list[tuple[float, float]],
) -> np.ndarray:
    if len(source_points) != 4 or len(target_points) != 4:
        raise ValueError("Homography requires exactly four source and target points")

    rows = []
    values = []
    for (src_x, src_y), (dst_x, dst_y) in zip(source_points, target_points):
        rows.append([src_x, src_y, 1.0, 0.0, 0.0, 0.0, -dst_x * src_x, -dst_x * src_y])
        values.append(dst_x)
        rows.append([0.0, 0.0, 0.0, src_x, src_y, 1.0, -dst_y * src_x, -dst_y * src_y])
        values.append(dst_y)

    coeffs = np.linalg.solve(
        np.array(rows, dtype=np.float64),
        np.array(values, dtype=np.float64),
    )
    a, b, c, d, e, f, g, h = coeffs
    return np.array(((a, b, c), (d, e, f), (g, h, 1.0))).T


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


class ProjectiveCoordConv:
    trans_mat: np.ndarray

    def __init__(
        self,
        source_points: list[tuple[float, float]],
        target_points: list[tuple[float, float]],
    ) -> None:
        self.trans_mat = _homography_from_points(source_points, target_points)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        x_, y_, z_ = np.array((x, y, 1.0)) @ self.trans_mat
        return x_ / z_, y_ / z_
