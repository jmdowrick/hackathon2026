"""Does conditioning on a future event bias the measured drift?

THE QUESTION THIS SETTLES
-------------------------
The pipeline learns H by matching the binned FIRST Kramers-Moyal moment -- the
mean displacement at each position. Path-sampling methods (TPS/TIS/RETIS) do
not bias the equations of motion, which makes them look like safe sources of
trajectory data. But they keep paths according to whether the path reaches a
target LATER. If conditioning on that future event tilts the displacement
distribution at intermediate positions, then feeding raw reactive paths into
the drift estimator gives the WRONG free energy, silently.

Theory says it does tilt, by the Doob h-transform: a diffusion conditioned to
reach B before A acquires an extra drift term

    b_reactive(x) = b_equilibrium(x) + 2 D grad ln q(x)

with q the committor. The correction diverges where q -> 0 and is largest
where ln q varies fastest -- i.e. right around the transition state, which is
exactly the region path sampling exists to visit.

This script measures all three quantities on the toy double well, where the
equilibrium drift is known in closed form and the committor is computed exactly
on the same generator the bond graph uses.

Run:  .venv/bin/python scripts/reactive_path_bias.py
"""

from __future__ import annotations

import numpy as np

from ala2bg import data, phnn
from ala2bg.grid import TorusGrid
from ala2bg.thermo import committor as cm
from ala2bg.thermo.smoluchowski import build_generator, rt

BARRIER = 12.0
D_TRUE = 0.01
DT = 0.5
TEMP = 298.0


def reactive_segments(phi, in_A, in_B):
    """Indices belonging to A->B reactive segments (last exit of A to first
    entry of B). This is what a transition path ensemble contains."""
    n = len(phi)
    state = np.zeros(n, dtype=np.int8)
    state[in_A] = -1
    state[in_B] = +1

    keep = np.zeros(n, dtype=bool)
    last_A = None
    for t in range(n):
        if state[t] == -1:
            last_A = t
        elif state[t] == +1 and last_A is not None:
            keep[last_A + 1:t] = True       # strictly between A and B
            last_A = None
    return keep


def main():
    print(__doc__)
    print("generating toy trajectories ...")
    d = data.simulate_toy(n_steps=400_000, n_traj=6, dt=DT, D=D_TRUE,
                          barrier=BARRIER, seed=7)
    X, tid = d["X"], d["traj_id"]
    grid = TorusGrid(40, 40)
    pts = grid.center_grid()

    # --- exact equilibrium drift (known analytically for the toy) ----------
    beta = 1.0 / rt(TEMP)
    b_eq = -D_TRUE * beta * data.toy_grad(pts, BARRIER)

    # --- exact committor on the generator the bond graph uses --------------
    F = data.toy_potential(pts, BARRIER)
    L, keep_idx = build_generator(F, D_TRUE, grid, TEMP)
    i_idx, _ = grid.unravel(keep_idx)
    # A = well near phi = -pi/2, B = well near phi = +pi/2
    phi_c = pts[keep_idx, 0]
    A = np.flatnonzero(np.abs(phi_c + np.pi / 2) < 0.25)
    B = np.flatnonzero(np.abs(phi_c - np.pi / 2) < 0.25)
    q_keep = cm.committor(L, A, B)
    q = np.full(grid.n_bins, np.nan)
    q[keep_idx] = q_keep

    # 2 D grad ln q on the grid (central differences, periodic)
    nphi, npsi = grid.shape
    lnq = np.log(np.clip(q, 1e-12, None)).reshape(nphi, npsi)
    h = grid.h
    dlnq = np.stack([
        ((np.roll(lnq, -1, 0) - np.roll(lnq, 1, 0)) / (2 * h[0])).ravel(),
        ((np.roll(lnq, -1, 1) - np.roll(lnq, 1, 1)) / (2 * h[1])).ravel(),
    ], axis=1)
    doob = 2.0 * D_TRUE * dlnq

    # --- measured drift: all frames vs reactive frames only ----------------
    in_A = np.abs(X[:, 0] + np.pi / 2) < 0.25
    in_B = np.abs(X[:, 0] - np.pi / 2) < 0.25

    react = np.zeros(len(X), dtype=bool)
    for t in np.unique(tid):
        sel = tid == t
        react[sel] = reactive_segments(X[sel, 0], in_A[sel], in_B[sel])

    print(f"frames total      {len(X):,}")
    print(f"frames reactive   {int(react.sum()):,} "
          f"({100 * react.mean():.2f}% -- this is what a TPE contains)")

    bd_all = phnn.binned_drift(X, DT, grid, tid, min_count=30)
    bd_rx = phnn.binned_drift(X[react], DT, grid, tid[react], min_count=30)

    m = bd_all["mask"] & bd_rx["mask"] & np.isfinite(q)
    # focus on the barrier region, where the correction is predicted to bite
    barrier_zone = m & (np.abs(pts[:, 0]) < 0.6)

    def rms(v):
        return float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))

    print()
    print("                                        all bins    barrier region")
    print(f"  bins compared                        {int(m.sum()):8d}    "
          f"{int(barrier_zone.sum()):10d}")
    print(f"  |drift_all - b_eq|          rms      "
          f"{rms((bd_all['drift'] - b_eq)[m]):8.5f}    "
          f"{rms((bd_all['drift'] - b_eq)[barrier_zone]):10.5f}")
    print(f"  |drift_reactive - b_eq|     rms      "
          f"{rms((bd_rx['drift'] - b_eq)[m]):8.5f}    "
          f"{rms((bd_rx['drift'] - b_eq)[barrier_zone]):10.5f}")
    print(f"  |drift_reactive - (b_eq+2D grad ln q)| "
          f"{rms((bd_rx['drift'] - b_eq - doob)[m]):6.5f}    "
          f"{rms((bd_rx['drift'] - b_eq - doob)[barrier_zone]):10.5f}")
    print(f"  |2D grad ln q| (the predicted tilt)   "
          f"{rms(doob[m]):8.5f}    {rms(doob[barrier_zone]):10.5f}")

    err_plain = rms((bd_rx["drift"] - b_eq)[barrier_zone])
    err_doob = rms((bd_rx["drift"] - b_eq - doob)[barrier_zone])
    err_unbiased = rms((bd_all["drift"] - b_eq)[barrier_zone])

    print()
    print(f"VERDICT (barrier region):")
    print(f"  unbiased data reproduces the equilibrium drift to  {err_unbiased:.5f}")
    print(f"  reactive-only data is off by                       {err_plain:.5f}"
          f"   ({err_plain / max(err_unbiased, 1e-12):.1f}x worse)")
    print(f"  after the Doob correction, off by                  {err_doob:.5f}")
    if err_plain > 2 * err_unbiased:
        print()
        print("  => Raw reactive paths DO bias the drift estimator.")
        print("     Transition-path data cannot be fed to binned_drift as-is;")
        print("     it needs per-frame weights that restore the equilibrium")
        print("     path measure (e.g. the Reweighted Path Ensemble).")
    else:
        print()
        print("  => No significant bias detected at this statistics level.")


if __name__ == "__main__":
    main()
