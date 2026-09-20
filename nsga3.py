"""Validated pymoo NSGA-III adapter with reference directions and run diagnostics."""

from dataclasses import dataclass
from math import comb

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import qmc


def nondominated_fronts(objectives, violations=None):
    f = np.asarray(objectives, dtype=float)
    cv = np.zeros(len(f)) if violations is None else np.asarray(violations)
    fronts = []
    for violation in np.unique(cv):
        indices = np.flatnonzero(cv == violation)
        fronts.extend(indices[front] for front in NonDominatedSorting().do(f[indices]))
    return fronts


@dataclass
class NSGAResult:
    points: np.ndarray
    objectives: np.ndarray
    violations: np.ndarray
    archive_points: np.ndarray
    archive_objectives: np.ndarray
    archive_violations: np.ndarray
    history: list
    reference_directions: np.ndarray


def minimize(objective, bounds, *, n_objectives=4, population_size=56, generations=32,
             reference_partitions=5, seed=7, initial_points=None, callback=None,
             duration_step=None, hv_reference=None):
    """Minimize finite objective vectors with nonnegative constraint violation.

    Uses Das-Dennis directions, normalized reference-direction survival, bounded
    SBX, polynomial mutation, duration repair, and duplicate elimination. Archive
    hypervolume uses fixed user-supplied units/scales, so generations are comparable.
    A flat recent hypervolume is a diagnostic only, not proof of convergence.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("bounds must be finite increasing pairs")
    if n_objectives < 2 or reference_partitions < 1 or generations < 1:
        raise ValueError("invalid NSGA-III dimensions/budget")
    count = comb(reference_partitions + n_objectives - 1, n_objectives - 1)
    if population_size < count:
        raise ValueError(f"population must cover all {count} reference directions")
    low, high = bounds.T
    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=reference_partitions)
    reference = np.ones(n_objectives) if hv_reference is None else np.asarray(hv_reference)
    if reference.shape != (n_objectives,) or np.any(reference <= 0) or not np.all(np.isfinite(reference)):
        raise ValueError("hypervolume reference must be finite and positive")
    initial = low + qmc.LatinHypercube(len(low), seed=seed).random(population_size)*(high-low)
    if initial_points is not None:
        supplied = np.atleast_2d(initial_points)
        if (supplied.shape[1] != len(low) or len(supplied) > population_size
                or not np.all(np.isfinite(supplied)) or np.any(supplied < low) or np.any(supplied > high)):
            raise ValueError("invalid initial_points")
        initial[:len(supplied)] = supplied
    archive_x, archive_f, archive_cv, history = [], [], [], []

    class ModelProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(low), n_obj=n_objectives, n_ieq_constr=1, xl=low, xu=high)

        def _evaluate(self, x, out, *args, **kwargs):
            f, cv = objective(x)
            f = np.asarray(f, dtype=float)
            if f.shape != (n_objectives,) or not np.all(np.isfinite(f)) or not np.isfinite(cv) or cv < 0:
                raise ValueError("invalid objective or constraint output")
            out["F"], out["G"] = f, [cv]
            archive_x.append(x.copy())
            archive_f.append(f.copy())
            archive_cv.append(float(cv))

    class GridRepair(Repair):
        def _do(self, problem, x, **kwargs):
            x = np.clip(x, low, high)
            if duration_step is not None:
                x[:, 0] = np.clip(np.rint(x[:, 0]/duration_step)*duration_step,
                                  np.ceil(low[0]/duration_step)*duration_step,
                                  np.floor(high[0]/duration_step)*duration_step)
            return x

    class Progress(Callback):
        def notify(self, algorithm):
            feasible = np.asarray(archive_cv) == 0
            f = np.asarray(archive_f)[feasible]
            hv = float(HV(ref_point=np.ones(n_objectives))(f/reference)) if len(f) else 0.0
            record = {"generation": int(algorithm.n_gen), "evaluations": len(archive_x),
                      "seed": seed, "feasible_count": int(np.sum(algorithm.pop.get("CV") <= 0)),
                      "pareto_count": len(NonDominatedSorting().do(f, only_non_dominated_front=True)) if len(f) else 0,
                      "archive_hypervolume": hv, "hypervolume_gain_last_10": None}
            if len(history) >= 10:
                previous = history[-10]["archive_hypervolume"]
                record["hypervolume_gain_last_10"] = (hv-previous)/max(abs(previous), 1e-12)
            history.append(record)
            if callback:
                callback(record)

    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=population_size, sampling=initial,
                      crossover=SBX(prob=.9, eta=15), mutation=PM(prob=1.0, eta=20),
                      repair=GridRepair(), eliminate_duplicates=True)
    result = pymoo_minimize(ModelProblem(), algorithm, ("n_gen", generations),
                            seed=seed, callback=Progress(), verbose=False)
    pop = result.pop
    return NSGAResult(pop.get("X"), pop.get("F"), pop.get("CV").ravel(),
                      np.asarray(archive_x), np.asarray(archive_f), np.asarray(archive_cv),
                      history, ref_dirs)
