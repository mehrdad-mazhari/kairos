"""Deterministic grid calibration for the optional legacy diagnostic scripts."""
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass
class GridResult:
    x: np.ndarray
    fun: float
    values: np.ndarray

    @property
    def best_so_far(self):
        return np.minimum.accumulate(self.values)


def grid_minimize(objective, bounds, *, x0, points_per_axis=21):
    points = np.vstack((x0, list(product(*(np.linspace(a, b, points_per_axis) for a, b in bounds)))))
    values = np.array([objective(x) for x in points])
    best = int(np.argmin(values))
    return GridResult(points[best], float(values[best]), values)
