"""Free-energy landscape + diffusion -> Markov generator, on a periodic grid.

The Bicout-Szabo discretisation of the Smoluchowski equation

    d(rho)/dt = div [ D ( grad rho + beta rho grad F ) ]

For neighbouring bins i, j at spacing h:

    k_{i->j} = (D_ij / h^2) * exp( -beta (F_j - F_i) / 2 )

This scheme is chosen because detailed balance is EXACT rather than approximate:

    k_{i->j} / k_{j->i} = exp( -beta (F_j - F_i) )

and therefore the stationary distribution is exactly pi_i ~ exp(-beta F_i).
That is what lets the bond graph downstream claim thermodynamic consistency as
a structural property rather than a fitted one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..grid import TorusGrid

# Gas constant in kJ/mol/K
R_GAS = 8.31446261815324e-3


def rt(temperature: float = 298.0) -> float:
    """RT in kJ/mol."""
    return R_GAS * temperature


def build_generator(
    F: np.ndarray,
    D: np.ndarray | float,
    grid: TorusGrid,
    temperature: float = 298.0,
    mask: np.ndarray | None = None,
):
    """Build the sparse generator L with dp/dt = L p.

    Parameters
    ----------
    F : (n_bins,) free energy in kJ/mol. NaN or masked bins are excluded.
    D : scalar, (n_bins,), or (n_bins, 2) diffusion in rad^2/ps.
        A (n_bins, 2) array is anisotropic-diagonal: column a is used for
        edges along axis a.
    mask : (n_bins,) bool, True for bins to keep. Unsampled bins must be
        dropped or they create spurious zero-free-energy sinks.

    Returns
    -------
    L : (n, n) csr_matrix, generator on the kept bins, L[j, i] = k_{i->j}
    keep : (n,) int array of kept flat bin indices
    """
    F = np.asarray(F, dtype=np.float64).ravel()
    n_bins = grid.n_bins
    if F.size != n_bins:
        raise ValueError(f"F has {F.size} entries, grid has {n_bins} bins")

    if mask is None:
        mask = np.isfinite(F)
    mask = np.asarray(mask, dtype=bool).ravel() & np.isfinite(F)

    keep = np.flatnonzero(mask)
    if keep.size == 0:
        raise ValueError("mask excludes every bin")
    remap = -np.ones(n_bins, dtype=int)
    remap[keep] = np.arange(keep.size)

    D = np.asarray(D, dtype=np.float64)
    if D.ndim == 0:
        Dfun = lambda a, b, axis: float(D)  # noqa: E731
    elif D.ndim == 1:
        Dfun = lambda a, b, axis: 0.5 * (D[a] + D[b])  # noqa: E731
    elif D.ndim == 2 and D.shape[1] == 2:
        Dfun = lambda a, b, axis: 0.5 * (D[a, axis] + D[b, axis])  # noqa: E731
    else:
        raise ValueError(f"unsupported D shape {D.shape}")

    beta = 1.0 / rt(temperature)
    h2 = grid.h ** 2

    rows, cols, vals = [], [], []
    out = np.zeros(keep.size)

    for a, b, axis in grid.neighbours():
        if not (mask[a] and mask[b]):
            continue
        ia, ib = remap[a], remap[b]
        dab = Dfun(a, b, axis)
        if dab <= 0:
            continue
        pref = dab / h2[axis]
        # k_{a->b} and k_{b->a}; detailed balance holds by construction
        k_ab = pref * np.exp(-0.5 * beta * (F[b] - F[a]))
        k_ba = pref * np.exp(-0.5 * beta * (F[a] - F[b]))

        rows.append(ib); cols.append(ia); vals.append(k_ab)
        rows.append(ia); cols.append(ib); vals.append(k_ba)
        out[ia] += k_ab
        out[ib] += k_ba

    n = keep.size
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L = L - sp.diags(out)
    L = L.tocsr()

    # Restrict to the largest connected component. Sparsely-sampled regions
    # leave isolated islands; keeping them makes the generator singular, so the
    # committor and MFPT solves fail (or silently return nonsense). Dropping
    # them is the honest move -- but report how much was dropped rather than
    # quietly shrinking the state space.
    n_comp, comp = sp.csgraph.connected_components(L, directed=False)
    if n_comp > 1:
        sizes = np.bincount(comp)
        biggest = int(np.argmax(sizes))
        sel = comp == biggest
        dropped = int((~sel).sum())
        keep = keep[sel]
        L = L[sel][:, sel]
        # rebuild diagonal so columns sum to zero on the restricted set
        L = L - sp.diags(np.asarray(L.sum(axis=0)).ravel())
        L.n_components_dropped = dropped  # noqa: B010  (informational)

    return L.tocsr(), keep


def stationary_from_F(F: np.ndarray, keep: np.ndarray, temperature: float = 298.0):
    """pi ~ exp(-beta F) on the kept bins, normalised."""
    beta = 1.0 / rt(temperature)
    f = np.asarray(F).ravel()[keep]
    w = np.exp(-beta * (f - f.min()))
    return w / w.sum()


def detailed_balance_residual(L, pi) -> float:
    """max |k_{i->j} pi_i - k_{j->i} pi_j| / scale.

    Should be at machine precision for a Bicout-Szabo generator. This is the
    numerical statement of 'detailed balance is structural'.
    """
    Lc = sp.coo_matrix(L)
    flux = {}
    for i, j, v in zip(Lc.row, Lc.col, Lc.data):
        if i == j:
            continue
        # L[i, j] = k_{j->i}
        flux[(j, i)] = v * pi[j]
    worst = 0.0
    scale = max(float(np.max(np.abs(list(flux.values())))) if flux else 1.0, 1e-300)
    for (a, b), f_ab in flux.items():
        f_ba = flux.get((b, a), 0.0)
        worst = max(worst, abs(f_ab - f_ba))
    return worst / scale
