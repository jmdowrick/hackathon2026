"""The learned model as a GENERATIVE SURROGATE -- the scalable route to rates.

WHY THIS EXISTS
---------------
The Bicout-Szabo route (smoluchowski.py -> bondgraph.py) needs a dense n^d
grid: it enumerates every bin as a Ce element and every edge as an Re. That is
fine at 2 CVs (3,600 states) and gone by 4-6 CVs. Every grid-based component
in this package inherits that wall -- the histogram FES, the grid selection,
the generator, the committor solve.

The learned model does not. H and D are continuous functions fitted on
OCCUPIED VOXELS only (nd.py), at O(n_frames) cost regardless of dimension. So
once they are fitted, the cheapest way to get kinetics is to SIMULATE THEM:

    expensive MD (sparse, high-dimensional, maybe microseconds of wall time)
        -> learn H, D                     dimension-free
        -> integrate the learned SDE      2-6 dims: free
        -> states + transition counts from the surrogate
        -> reversible rate matrix -> Ce/Re -> CellML

Integrating a 4-dimensional overdamped Langevin equation is trivial next to a
100k-atom bilayer, so the surrogate supplies ARBITRARILY GOOD statistics in
exactly the region where the real trajectories are thinnest -- the barrier.
That attacks the measured failure mode (an empty barrier bin disconnects the
generator and silently deletes the slow process) structurally, rather than by
sampling the real system harder.

It also sidesteps the continuous-time embedding problem: there is no matrix
logarithm, because the lag time is a free choice on data we can generate
without limit.

WHAT IT IS NOT
--------------
It is not new physics, and it cannot invent information the fit did not
contain. If H is wrong in a region the real MD never visited, the surrogate
will be confidently wrong there. That is why validate_against() compares
surrogate rates with rates measured directly from the real trajectories: the
surrogate never sees transition counts, only local drift, so agreement is a
genuine test rather than a restatement.
"""

from __future__ import annotations

import numpy as np

from .cvs import wrap, wrapped_delta
from .thermo.smoluchowski import rt


def simulate_learned(model, D_const, n_walkers=2000, n_steps=20_000, dt=1.0,
                     temperature=298.0, dim=2, seed=0, device=None,
                     x0=None, stride=1):
    """Integrate xdot = -D beta grad H(x) + sqrt(2D) eta on the torus.

    All walkers advance together as one batched tensor, so the cost is one
    network gradient evaluation per step regardless of walker count. Returns
    (X, traj_id) in the same contract every other estimator consumes.

    dt here is the SURROGATE timestep in the same time unit as the D used to
    fit the model (ps if D is rad^2/ps). It need not equal the MD frame
    spacing -- it only has to be small enough for Euler-Maruyama.
    """
    import torch

    device = device or next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    beta = 1.0 / rt(temperature)
    sig = float(np.sqrt(2.0 * D_const * dt))

    if x0 is None:
        x = (torch.rand((n_walkers, dim), generator=g) * 2 * np.pi - np.pi)
    else:
        x = torch.tensor(np.asarray(x0), dtype=torch.float32)
    x = x.to(device)

    n_save = n_steps // stride
    out = np.empty((n_save, n_walkers, dim), dtype=np.float32)

    k = 0
    for s in range(n_steps):
        xr = x.detach().requires_grad_(True)
        h = model(xr).sum()
        (grad,) = torch.autograd.grad(h, xr)
        noise = torch.randn(x.shape, generator=g).to(device)
        x = x - D_const * beta * grad * dt + sig * noise
        # wrap onto the torus
        x = (x + np.pi) % (2 * np.pi) - np.pi
        x = x.detach()
        if (s % stride) == 0 and k < n_save:
            out[k] = x.cpu().numpy()
            k += 1

    out = out[:k]
    X = np.concatenate([out[:, w, :] for w in range(n_walkers)]).astype(np.float64)
    tid = np.concatenate([np.full(len(out), w) for w in range(n_walkers)])
    return wrap(X), tid


def assign_states(X, centers):
    """Nearest-centre assignment on the torus. Centres define the states --
    a handful of them, never n^d bins."""
    X = np.asarray(X)
    d = np.stack([
        np.linalg.norm(wrapped_delta(X, c[None, :]), axis=1) for c in centers
    ], axis=1)
    return np.argmin(d, axis=1)


def rate_matrix_from_counts(labels, traj_id, lag, dt, n_states,
                            temperature=298.0):
    """Reversible rate matrix from transition counts at the given lag.

    Detailed balance is imposed by symmetrising the COUNT matrix (the standard
    reversible estimator in the maximum-likelihood-with-symmetric-counts
    sense), which is what the bond graph requires: the Wegscheider condition
    IS detailed balance.

    No matrix logarithm is taken. With a surrogate we can simply use a lag
    short enough that k ~ C / (tau * n_i) is a good generator estimate, and
    verify by checking the result is lag-insensitive.
    """
    labels = np.asarray(labels)
    traj_id = np.asarray(traj_id)
    ok = traj_id[:-lag] == traj_id[lag:]
    a = labels[:-lag][ok]
    b = labels[lag:][ok]

    C = np.zeros((n_states, n_states))
    np.add.at(C, (a, b), 1.0)
    Csym = 0.5 * (C + C.T)                 # enforce detailed balance

    n_i = Csym.sum(axis=1)
    pi = n_i / max(n_i.sum(), 1.0)

    tau = lag * dt
    K = np.zeros((n_states, n_states))
    nz = n_i > 0
    K[nz] = Csym[nz] / (n_i[nz, None] * tau)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(K, -K.sum(axis=1))
    return K, pi, Csym


def mfpt_from_rates(K, A, B):
    """MFPT from state set A to set B on a small rate matrix (dense solve)."""
    import scipy.linalg as sla

    n = K.shape[0]
    inB = np.zeros(n, dtype=bool)
    inB[np.asarray(B, dtype=int)] = True
    T = ~inB
    if not T.any():
        return float("nan")
    tau = np.zeros(n)
    try:
        tau[T] = sla.solve(K[np.ix_(T, T)], -np.ones(T.sum()))
    except Exception:
        return float("nan")
    A = np.asarray(A, dtype=int)
    return float(np.mean(tau[A]))


def timescales_from_rates(K, k=4):
    vals = np.linalg.eigvals(K)
    vals = np.sort(np.real(vals))[::-1]
    vals = vals[vals < -1e-14]
    if vals.size == 0:
        return np.array([])
    return np.sort(-1.0 / vals)[::-1][:k]


def validate_against(real_X, real_tid, real_dt, surr_X, surr_tid, surr_dt,
                     centers, lag_real=1, lag_surr=1, temperature=298.0):
    """Compare surrogate kinetics with kinetics measured from the real data.

    This is the test that matters. The surrogate was fitted on LOCAL DRIFT
    only -- it never saw a transition count, a rate, or a timescale. So if its
    rates reproduce the ones counted directly from the real trajectories, the
    learned landscape carries real kinetic information rather than echoing
    something it was told.
    """
    n = len(centers)
    lab_r = assign_states(real_X, centers)
    lab_s = assign_states(surr_X, centers)

    K_r, pi_r, C_r = rate_matrix_from_counts(lab_r, real_tid, lag_real,
                                             real_dt, n, temperature)
    K_s, pi_s, C_s = rate_matrix_from_counts(lab_s, surr_tid, lag_surr,
                                             surr_dt, n, temperature)

    ts_r = timescales_from_rates(K_r)
    ts_s = timescales_from_rates(K_s)

    beta = 1.0 / rt(temperature)
    with np.errstate(divide="ignore"):
        F_r = -np.log(np.clip(pi_r, 1e-300, None)) / beta
        F_s = -np.log(np.clip(pi_s, 1e-300, None)) / beta
    F_r -= F_r.min(); F_s -= F_s.min()

    out = {
        "n_states": n,
        "pi_real": pi_r.tolist(), "pi_surrogate": pi_s.tolist(),
        "F_real_kJ": F_r.tolist(), "F_surrogate_kJ": F_s.tolist(),
        "F_rmse_kJ": float(np.sqrt(np.mean((F_r - F_s) ** 2))),
        "timescales_real_ps": ts_r.tolist(),
        "timescales_surrogate_ps": ts_s.tolist(),
        "counts_real": int(C_r.sum()), "counts_surrogate": int(C_s.sum()),
    }
    if ts_r.size and ts_s.size:
        out["slowest_ratio"] = float(max(ts_r[0], ts_s[0]) /
                                     max(min(ts_r[0], ts_s[0]), 1e-30))
    return out, K_r, K_s


def bondgraph_from_rates(K, pi, temperature=298.0, labels=None, volume=1.0):
    """Rate matrix + stationary distribution -> Ce/Re bond graph. No grid.

    K_i    = exp(beta F_i) / V   with  F_i = -RT ln pi_i
    kappa_ij = k_{i->j} / K_i

    Detailed balance in the rate matrix IS the Wegscheider condition, so the
    resulting kappa is symmetric by construction -- the same guarantee the
    grid route gets from Bicout-Szabo, obtained here without any grid.

    The reference volume V is arbitrary and cancels in every ratio and in
    every rate; it is recorded so the absolute K values can be interpreted.
    """
    from .thermo.bondgraph import BondGraph

    pi = np.asarray(pi, dtype=float)
    n = len(pi)
    beta = 1.0 / rt(temperature)
    F = -np.log(np.clip(pi, 1e-300, None)) / beta
    F = F - F.min()
    Kc = np.exp(beta * F) / volume

    edges, kappas = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] <= 0 or K[j, i] <= 0:
                continue
            kap_ij = K[i, j] / Kc[i]
            kap_ji = K[j, i] / Kc[j]
            rel = abs(kap_ij - kap_ji) / max(abs(kap_ij), abs(kap_ji), 1e-300)
            if rel > 1e-6:
                raise AssertionError(
                    f"kappa asymmetric on edge ({i},{j}): rel {rel:.2e}. "
                    "The rate matrix does not satisfy detailed balance."
                )
            edges.append((i, j))
            kappas.append(0.5 * (kap_ij + kap_ji))

    return BondGraph(
        K=Kc, kappa=np.array(kappas), edges=np.array(edges, dtype=int),
        F=F, temperature=temperature, volumes=np.full(n, volume),
        labels=labels,
        meta={"route": "surrogate", "reference_volume": volume,
              "note": "states are clusters, not grid bins; the reference "
                      "volume is arbitrary and cancels in all ratios"},
    )
