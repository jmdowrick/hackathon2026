"""Measure D from the second Kramers-Moyal moment. MEASURED, NEVER LEARNED.

    D(x) = < dx dx^T > / (2 dt)

The point: this estimator does NOT require knowing H. That is what breaks the
H/D degeneracy described in phnn.py -- measure D first, freeze it, and H is
then exactly identified from the first moment.

Two rungs:
  constant  -- D = global mean. Then div D = 0 and the Ito spurious-drift term
               vanishes identically. Default, most robust.
  binned    -- D(x) per grid bin. Better, because what actually sets a rate is
               D AT THE BARRIER TOP, which is precisely where a constant-D
               assumption costs the most. Requires carrying div D (which much
               of the ML literature silently drops -- it is wrong to do so).
"""

from __future__ import annotations

import numpy as np

from .cvs import wrapped_delta
from .grid import TorusGrid


def measure_D(X: np.ndarray, dt: float, traj_id: np.ndarray | None = None,
              lag: int = 1) -> dict:
    """Global (constant, isotropic) D from short-time displacements.

    Returns per-dimension and isotropic estimates in rad^2 / (time unit of dt).
    """
    X = np.asarray(X, dtype=np.float64)
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)

    ok = traj_id[:-lag] == traj_id[lag:]
    dx = wrapped_delta(X[lag:][ok], X[:-lag][ok])
    tau = dt * lag

    per_dim = np.mean(dx ** 2, axis=0) / (2.0 * tau)
    return {
        "D_per_dim": per_dim,
        "D_const": float(np.mean(per_dim)),
        "lag": lag,
        "tau": tau,
        "n_pairs": int(len(dx)),
    }


def lag_stability(X, dt, traj_id=None, lags=(1, 2, 4)) -> dict:
    """D across several lags.

    A genuinely diffusive coordinate gives a lag-independent D. Drift with lag
    is the signature of non-Markovian behaviour in the chosen CVs -- useful
    information, not a nuisance, and it is reported rather than hidden.
    """
    out = {}
    for L in lags:
        try:
            out[int(L)] = float(measure_D(X, dt, traj_id, lag=L)["D_const"])
        except Exception:
            out[int(L)] = float("nan")
    vals = np.array([v for v in out.values() if np.isfinite(v)])
    spread = float(vals.max() / vals.min()) if vals.size > 1 and vals.min() > 0 else float("nan")
    return {"by_lag": out, "max_ratio": spread}


def memory_time(X, dt, traj_id, D_measured, timescale_model, timescale_ref,
                lags=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)):
    """Turn a timescale disagreement into a measurement of memory time.

    An overdamped Langevin model in these CVs gives relaxation times inversely
    proportional to D. If the model (built with the short-time D) is faster
    than an independent reference (the MSM), then the D that would reconcile
    them is

        D_eff = D_measured * t_model / t_reference

    On a genuinely Markovian coordinate D is lag-independent and D_eff equals
    D_measured. On a coordinate with memory, D falls with lag, and the lag at
    which the measured curve passes through D_eff is an estimate of the
    correlation time being neglected.

    This is the honest reading of the discrepancy: not "the model is wrong by
    17x", but "an overdamped model in (phi, psi) needs an effective diffusivity
    measured on a ~X ps lag, which is the memory time of these coordinates".
    """
    if not (timescale_model and timescale_ref) or timescale_ref <= 0:
        return {"available": False}

    d_eff = D_measured * (timescale_model / timescale_ref)
    curve = {}
    for L in lags:
        try:
            curve[int(L)] = float(measure_D(X, dt, traj_id, lag=int(L))["D_const"])
        except Exception:
            continue

    lag_match, prev_l, prev_v = None, None, None
    for L, v in sorted(curve.items()):
        if prev_v is not None and (prev_v - d_eff) * (v - d_eff) <= 0:
            # linear interpolation in log D vs log lag
            if v != prev_v:
                f = (np.log(d_eff) - np.log(prev_v)) / (np.log(v) - np.log(prev_v))
                lag_match = float(np.exp(np.log(prev_l) + f * (np.log(L) - np.log(prev_l))))
            else:
                lag_match = float(L)
            break
        prev_l, prev_v = L, v

    return {
        "available": True,
        "D_measured_lag1": float(D_measured),
        "D_eff_to_reconcile": float(d_eff),
        "ratio": float(D_measured / d_eff) if d_eff > 0 else float("nan"),
        "D_vs_lag_ps": curve,
        "memory_time_ps": lag_match,
        "interpretation": (
            "D is lag-independent for a Markovian coordinate. Here it is not, "
            "so the short-time D overestimates the effective diffusivity for "
            "barrier crossing. The lag at which measured D equals D_eff "
            "estimates the memory time neglected by the overdamped model."
            if lag_match else
            "no lag in the scanned range reproduces D_eff; the discrepancy is "
            "not explained by the lag dependence of D alone."
        ),
    }


def measure_D_binned(X, dt, grid: TorusGrid, traj_id=None, lag: int = 1,
                     min_count: int = 20) -> dict:
    """Per-bin D(x). Bins with too few samples fall back to the global value."""
    X = np.asarray(X, dtype=np.float64)
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)

    ok = traj_id[:-lag] == traj_id[lag:]
    x0 = X[:-lag][ok]
    dx = wrapped_delta(X[lag:][ok], X[:-lag][ok])
    tau = dt * lag

    idx = grid.index_of(x0)
    n = grid.n_bins
    counts = np.bincount(idx, minlength=n)
    D = np.zeros((n, 2))
    for a in range(2):
        s = np.bincount(idx, weights=dx[:, a] ** 2, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[:, a] = s / (2.0 * tau * counts)

    glob = measure_D(X, dt, traj_id, lag)["D_per_dim"]
    thin = counts < min_count
    D[thin, 0] = glob[0]
    D[thin, 1] = glob[1]
    D[~np.isfinite(D)] = float(np.mean(glob))

    return {
        "D": D, "counts": counts, "n_thin": int(thin.sum()),
        "lag": lag, "tau": tau, "D_global": glob,
    }


def divergence(D_binned: np.ndarray, grid: TorusGrid) -> np.ndarray:
    """div D on the periodic grid, central differences.

    Needed for the drift when D is position-dependent. Dropping this term is a
    real (and common) error, not a harmless simplification.
    """
    n_phi, n_psi = grid.shape
    Dphi = D_binned[:, 0].reshape(n_phi, n_psi)
    Dpsi = D_binned[:, 1].reshape(n_phi, n_psi)
    h = grid.h
    d_dphi = (np.roll(Dphi, -1, axis=0) - np.roll(Dphi, 1, axis=0)) / (2 * h[0])
    d_dpsi = (np.roll(Dpsi, -1, axis=1) - np.roll(Dpsi, 1, axis=1)) / (2 * h[1])
    return np.stack([d_dphi.ravel(), d_dpsi.ravel()], axis=1)
