"""Can a state-dependent bias (REST / metadynamics / umbrella) feed this pipeline?

THE CLAIM UNDER TEST
--------------------
Two kinds of enhanced sampling affect the drift estimator completely
differently, and the distinction is NOT "biased vs unbiased dynamics":

  STATE-DEPENDENT BIAS (REST2/gREST, metadynamics, umbrella):
      simulate on V' = V + U.  The drift becomes
          b'(x) = -D beta grad(V + U) = b(x) - D beta grad U(x)
      so the true drift is recovered by ADDING THE BIAS FORCE BACK:
          b(x) = b'(x) + D beta grad U(x)
      an analytic, LOCAL correction requiring no path reweighting.
      D itself is unchanged: for Langevin dynamics the noise amplitude is set
      by the thermostat (T, gamma), not by the potential.

  FUTURE-CONDITIONED SELECTION (TPS / TIS / RETIS):
      the conditional displacement distribution is itself tilted
      (Doob h-transform, b_reactive = b_eq + 2 D grad ln q), and undoing it
      needs the committor you were trying to compute.
      See scripts/reactive_path_bias.py -- measured 3.7x bias.

WHY THE ESTIMATOR CARES ABOUT THIS DISTINCTION
----------------------------------------------
The pipeline fits CONDITIONAL moments at a position. A conditional moment does
not depend on how OFTEN x is visited, only on what happens once you are there.
So the sampling weight is irrelevant to it -- which is exactly why enhanced
sampling that only reshapes WHERE you go is recoverable, while selection that
depends on WHERE YOU WILL GO is not.

That is the scalability argument: as systems grow, the binding constraint is
coverage of the barrier region (an empty barrier bin disconnects the generator
and silently deletes the slow process -- see gridselect.py). A state-dependent
bias fixes coverage AND stays correctable.

Run:  .venv/bin/python scripts/biased_drift_recovery.py
"""

from __future__ import annotations

import numpy as np

from ala2bg import diffusion, fes, phnn
from ala2bg.cvs import wrap
from ala2bg.data import toy_grad, toy_potential
from ala2bg.grid import TorusGrid
from ala2bg.thermo.smoluchowski import rt

BARRIER = 14.0
D_TRUE = 0.01
DT = 0.5
TEMP = 298.0


def bias_potential(x, strength):
    """U(phi) that flattens the barrier -- a stand-in for REST2 scaling the
    solute terms, or for a converged metadynamics bias."""
    x = np.atleast_2d(x)
    return -0.5 * strength * (1.0 + np.cos(2.0 * x[:, 0]))


def bias_grad(x, strength):
    x = np.atleast_2d(x)
    return np.stack([strength * np.sin(2.0 * x[:, 0]), np.zeros(len(x))], axis=1)


def simulate(strength, n_steps=150_000, n_traj=8, seed=0):
    """Brownian dynamics on V + U. strength=0 is the unbiased reference."""
    rng = np.random.default_rng(seed)
    beta = 1.0 / rt(TEMP)
    sig = np.sqrt(2.0 * D_TRUE * DT)
    x = wrap(rng.uniform(-np.pi, np.pi, size=(n_traj, 2)))
    out = np.empty((n_steps, n_traj, 2))
    for s in range(n_steps):
        g = toy_grad(x, BARRIER) + bias_grad(x, strength)
        x = wrap(x - D_TRUE * beta * g * DT + sig * rng.standard_normal((n_traj, 2)))
        out[s] = x
    X = np.concatenate([out[:, t, :] for t in range(n_traj)])
    tid = np.concatenate([np.full(n_steps, t) for t in range(n_traj)])
    return X, tid


def fit(X, tid, grid, correction=None, min_count=25):
    """Binned drift -> optional analytic correction -> least-squares H."""
    bd = phnn.binned_drift(X, DT, grid, tid, min_count=min_count)
    drift = bd["drift"].copy()
    if correction is not None:
        drift = drift + correction
    m = bd["mask"]

    # Recover H from grad H by solving the Poisson equation on the torus:
    #     lap H = div(target),   target = grad H = -b / (D beta)
    # On a periodic grid the FFT solves this exactly and instantly, with no
    # step size to tune and nothing to diverge. The k=0 mode is the arbitrary
    # additive constant and is set to zero.
    beta = 1.0 / rt(TEMP)
    target = -drift / (D_TRUE * beta)
    nphi, npsi = grid.shape
    h = grid.h

    tphi = np.where(m, target[:, 0], 0.0).reshape(nphi, npsi)
    tpsi = np.where(m, target[:, 1], 0.0).reshape(nphi, npsi)
    div = ((np.roll(tphi, -1, 0) - np.roll(tphi, 1, 0)) / (2 * h[0])
           + (np.roll(tpsi, -1, 1) - np.roll(tpsi, 1, 1)) / (2 * h[1]))

    kphi = 2 * np.pi * np.fft.fftfreq(nphi, d=h[0])
    kpsi = 2 * np.pi * np.fft.fftfreq(npsi, d=h[1])
    K2 = kphi[:, None] ** 2 + kpsi[None, :] ** 2
    K2[0, 0] = 1.0                              # avoid 0/0; mode set below
    Hk = -np.fft.fft2(div) / K2
    Hk[0, 0] = 0.0
    H = np.real(np.fft.ifft2(Hk)).ravel()
    H = H - H[m].mean()
    return H, m


def main():
    print(__doc__)
    grid = TorusGrid(30, 30)
    pts = grid.center_grid()
    F_true = toy_potential(pts, BARRIER); F_true -= F_true.min()
    beta = 1.0 / rt(TEMP)

    rows = []
    for strength, label in [(0.0, "unbiased (reference)"),
                            (10.0, "biased, UNCORRECTED"),
                            (10.0, "biased, force-corrected")]:
        X, tid = simulate(strength, seed=3)
        corr = None
        if "force-corrected" in label:
            # add the known bias force back: b = b' + D beta grad U
            corr = D_TRUE * beta * bias_grad(pts, strength)

        Dm = diffusion.measure_D(X, DT, tid, lag=1)["D_const"]
        _, counts, _ = fes.histogram_fes(X, grid, min_count=1)
        barrier_counts = int(counts[np.abs(pts[:, 0]) < 0.4].sum())

        H, m = fit(X, tid, grid, corr)
        sel = m & (counts >= 25)
        hh = H[sel] - H[sel].mean(); ff = F_true[sel] - F_true[sel].mean()
        rmse = float(np.sqrt(np.mean((hh - ff) ** 2)))
        corrcoef = float(np.corrcoef(hh, ff)[0, 1])
        rows.append((label, Dm, barrier_counts, int(sel.sum()), rmse, corrcoef))

    print(f"{'case':26s} {'D_meas':>8s} {'barrier pts':>12s} "
          f"{'bins':>6s} {'RMSE kJ/mol':>12s} {'corr':>7s}")
    for label, Dm, bc, nb, rmse, cc in rows:
        print(f"{label:26s} {Dm:8.5f} {bc:12d} {nb:6d} {rmse:12.2f} {cc:7.4f}")

    print()
    print(f"true D = {D_TRUE}")
    ref, unc, cor = rows
    print(f"coverage of the barrier region improved "
          f"{unc[2] / max(ref[2], 1):.1f}x under bias")
    print(f"D is unchanged by the bias: {ref[1]:.5f} -> {unc[1]:.5f} "
          f"(thermostat sets the noise, not the potential)")
    print()
    print("VERDICT")
    print(f"  uncorrected biased fit   RMSE {unc[4]:.2f} kJ/mol  corr {unc[5]:.4f}")
    print(f"  force-corrected fit      RMSE {cor[4]:.2f} kJ/mol  corr {cor[5]:.4f}")
    print(f"  unbiased reference       RMSE {ref[4]:.2f} kJ/mol  corr {ref[5]:.4f}")
    if cor[4] < 0.5 * unc[4]:
        print()
        print("  => A state-dependent bias IS recoverable analytically.")
        print("     No Girsanov path reweighting needed for the short-lag")
        print("     Kramers-Moyal estimators: just add the bias force back.")
        print("     (Girsanov IS needed for finite-lag path quantities such as")
        print("      the MSM reference arm, where the correction is non-local.)")
    else:
        print()
        print("  => correction did not recover the landscape; claim not supported.")


if __name__ == "__main__":
    main()
