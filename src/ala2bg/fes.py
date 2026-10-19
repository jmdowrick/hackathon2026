"""Empirical free energy surface by histogram, and basin assignment.

F = -RT ln P. With unbiased trajectories there is no reweighting to do and no
bias to correct -- which is exactly why the project can skip enhanced sampling
on the critical path.

This F is the THERMODYNAMIC route to the landscape. The PHNN learns H from the
drift, which is the KINETIC route. Nothing forces them to agree, so comparing
them tests whether (phi, psi) is a good Markovian coordinate.
"""

from __future__ import annotations

import numpy as np

from .grid import TorusGrid
from .thermo.smoluchowski import rt

# Reference basin centres in DEGREES. Stable to ~+-15 deg across force fields;
# use as assignment seeds and sanity checks, never as a quantitative metric.
BASINS_DEG = {
    "C7eq":   (-80.0,  75.0),
    "alphaR": (-70.0, -25.0),
    "C7ax":   (75.0,  -75.0),
    "alphaL": (50.0,   30.0),
}


def histogram_fes(X: np.ndarray, grid: TorusGrid, temperature: float = 298.0,
                  min_count: int = 1):
    """F in kJ/mol on the grid, plus counts and a sampled-bin mask."""
    X = np.asarray(X, dtype=np.float64)
    idx = grid.index_of(X)
    counts = np.bincount(idx, minlength=grid.n_bins).astype(float)
    mask = counts >= min_count

    p = np.where(mask, counts, np.nan)
    p = p / np.nansum(p)
    F = np.full(grid.n_bins, np.nan)
    F[mask] = -rt(temperature) * np.log(p[mask])
    if mask.any():
        F[mask] -= np.nanmin(F[mask])
    return F, counts, mask


def bootstrap_fes_error(X, grid, temperature=298.0, n_boot=20, traj_id=None,
                        seed=0, min_count=1):
    """Bootstrap error on F, resampling WHOLE TRAJECTORIES.

    Resampling frames would badly understate the error because consecutive
    frames are correlated; trajectories are the independent unit.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)
    uniq = np.unique(traj_id)

    acc = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.flatnonzero(traj_id == t) for t in pick])
        F, _, _ = histogram_fes(X[sel], grid, temperature, min_count)
        acc.append(F)
    A = np.array(acc)
    return np.nanstd(A, axis=0)


def assign_basins(grid: TorusGrid, mask: np.ndarray | None = None,
                  basins: dict | None = None) -> tuple:
    """Nearest-reference-centre assignment on the torus.

    Deliberately simple and deterministic: a watershed would be prettier but
    this is reproducible, explicable in one sentence, and good enough to define
    the coarse states for the CellML model.
    """
    basins = basins or BASINS_DEG
    names = list(basins)
    centres = np.radians(np.array([basins[k] for k in names]))
    pts = grid.center_grid()

    d = np.zeros((len(pts), len(names)))
    for k, c in enumerate(centres):
        delta = (pts - c + np.pi) % (2 * np.pi) - np.pi
        d[:, k] = np.linalg.norm(delta, axis=1)
    lab = np.argmin(d, axis=1)
    if mask is not None:
        lab = np.where(mask, lab, -1)
    return lab, names


def basin_free_energies(F, counts, labels, names, temperature=298.0) -> dict:
    """Basin free energies from summed populations, relative to the deepest."""
    out = {}
    tot = counts.sum()
    for k, nm in enumerate(names):
        sel = labels == k
        pop = counts[sel].sum()
        out[nm] = (-rt(temperature) * np.log(pop / tot)) if pop > 0 else float("nan")
    finite = [v for v in out.values() if np.isfinite(v)]
    if finite:
        lo = min(finite)
        out = {k: (v - lo if np.isfinite(v) else v) for k, v in out.items()}
    return out


def count_transitions(X, grid, labels, names, traj_id=None) -> dict:
    """Direct basin-to-basin transition counts -- the third, model-free arm.

    This is 'what the data plainly shows', independent of both the MSM's
    discretisation and the PHNN's fit.
    """
    X = np.asarray(X)
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)

    lab = labels[grid.index_of(X)]
    counts = {}
    cur, start = None, None
    for t in range(len(lab)):
        if t > 0 and traj_id[t] != traj_id[t - 1]:
            cur, start = None, None
        b = lab[t]
        if b < 0:
            continue
        if cur is None:
            cur, start = b, t
        elif b != cur:
            key = f"{names[cur]}->{names[b]}"
            counts.setdefault(key, {"n": 0, "dwell": []})
            counts[key]["n"] += 1
            counts[key]["dwell"].append(t - start)
            cur, start = b, t
    return {k: {"n": v["n"], "mean_dwell_frames": float(np.mean(v["dwell"]))}
            for k, v in counts.items()}
