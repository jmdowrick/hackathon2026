"""Transition-state descriptors: the features that actually govern kappa.

WHY THIS MODULE EXISTS
----------------------
phgnn.py established, analytically and numerically, that

    kappa_ij = V exp(-beta F_i) / MFPT(i->j),   MFPT ~ (1/D) exp(beta (F+ - F_i))
    =>  ln kappa_ij ~ ln D - beta F+

so F_i cancels and kappa is governed by the SADDLE F+, not by the endpoint
free energies. Measured over 260 edges: the saddle form reaches R2 = 0.871
while endpoint free energies reach 0.793, and the two correlate -0.80 -- so a
cheap-feature surrogate only works through a correlation that is a property of
the system, not a law.

The constructive consequence: a rate surrogate needs features ABOUT THE
BARRIER. This module computes them from a landscape -- learned or empirical --
so that kappa becomes predictable WITHOUT a path-sampling campaign, provided
the landscape itself is good enough in the transition region.

That proviso is the whole question, and test_regressions tests it directly:
the barrier is read off a LEARNED H and the resulting kappa compared with the
exact answer.

WHAT IS COMPUTED
----------------
  minimax_barrier  the lowest F at which two basins become connected, i.e. the
                   saddle on the minimum-energy path. Exact on the grid, via a
                   single sort plus union-find: O(N log N), no path guessing,
                   no initial guess, and it finds the true minimax path rather
                   than a straight-line interpolation.

  committor_barrier  the free energy of the q = 0.5 isoband. This is the
                   transition-state ensemble in the proper (committor) sense
                   rather than the geometric one, and it is the quantity
                   Gen-COMPAS uses a neural network to approximate.
"""

from __future__ import annotations

import numpy as np

from .thermo.smoluchowski import rt


def _neighbour_table(grid):
    nb = {}
    for a, b, _ in grid.neighbours():
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    return nb


def minimax_barrier(F, grid, centres, nb=None):
    """Saddle height between every pair of basins, by flooding.

    Sort bins by increasing F and union neighbours as they activate; the F at
    which two basins' components first merge is exactly the minimax barrier
    between them -- the highest point on the lowest path.

    F : (n_bins,) free energy, NaN allowed for unsampled bins
    centres : flat bin indices of the basin minima
    returns (n, n) array of saddle free energies, NaN where never connected
    """
    F = np.asarray(F, dtype=float).ravel()
    n = len(centres)
    sad = np.full((n, n), np.nan)
    if n < 2:
        return sad

    nb = nb or _neighbour_table(grid)
    finite = np.isfinite(F)
    order = np.argsort(np.where(finite, F, np.inf))
    order = order[finite[order]]

    parent = np.arange(grid.n_bins)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    basin_of = {int(c): i for i, c in enumerate(centres)}
    comp = {}                      # root -> set of basins it contains
    active = np.zeros(grid.n_bins, dtype=bool)

    for idx in order:
        idx = int(idx)
        active[idx] = True
        r = find(idx)
        comp.setdefault(r, set())
        if idx in basin_of:
            comp[r].add(basin_of[idx])

        for j in nb.get(idx, []):
            if not active[j]:
                continue
            ra, rb = find(idx), find(j)
            if ra == rb:
                continue
            sa, sb = comp.get(ra, set()), comp.get(rb, set())
            if sa and sb:
                for x in sa:
                    for y in sb:
                        if np.isnan(sad[x, y]):
                            # the merge happens at the current flood level
                            sad[x, y] = sad[y, x] = F[idx]
            parent[ra] = rb
            comp[rb] = sa | sb
            comp.pop(ra, None)

    return sad


def basins_from_landscape(F, grid, max_states=8, min_separation=0.5):
    """Local minima of F on the periodic grid, deepest first.

    min_separation (radians) merges minima that are too close to be distinct
    states, which otherwise inflates the state count with numerical noise.
    """
    F = np.asarray(F, dtype=float)
    n1, n2 = grid.shape
    Fg = np.where(np.isfinite(F), F, np.inf).reshape(n1, n2)
    is_min = np.ones_like(Fg, dtype=bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            is_min &= Fg <= np.roll(np.roll(Fg, a, 0), b, 1)
    is_min &= np.isfinite(Fg)

    idx = np.flatnonzero(is_min.ravel())
    idx = idx[np.argsort(F[idx])]

    pts = grid.center_grid()
    kept = []
    for i in idx:
        if len(kept) >= max_states:
            break
        if all(np.linalg.norm((pts[i] - pts[k] + np.pi) % (2 * np.pi) - np.pi)
               > min_separation for k in kept):
            kept.append(int(i))
    return np.array(kept, dtype=int)


def committor_barrier(L, keep, F, A_idx, B_idx, band=0.05, temperature=298.0):
    """Free energy of the committor q = 0.5 +- band isoband.

    The transition state in the proper sense: the set the system is equally
    likely to leave towards either basin. Gen-COMPAS learns this with a neural
    network because its space is 3N Cartesian coordinates; on a CV grid it is
    a sparse linear solve.
    """
    from .thermo import committor as cm

    q = cm.committor(L, A_idx, B_idx)
    sel = np.abs(q - 0.5) <= band
    if not sel.any():
        return {"available": False, "note": "no bins in the q=0.5 isoband"}

    f = np.asarray(F).ravel()[keep][sel]
    beta = 1.0 / rt(temperature)
    # Boltzmann-weighted free energy of the isoband
    w = np.exp(-beta * (f - f.min()))
    F_ts = f.min() - rt(temperature) * np.log(w.sum() / max(len(f), 1))
    return {
        "available": True,
        "F_ts_kJ": float(F_ts),
        "F_ts_min_kJ": float(f.min()),
        "n_bins": int(sel.sum()),
        "band": band,
    }


def kappa_from_barrier(F_saddle, D, cell_area, h, temperature=298.0):
    """The analytic Re constant implied by a barrier height.

    From the Bicout-Szabo form, kappa_ij = (D V / h^2) exp(-beta (F_i + F_j)/2)
    at the saddle, which for a barrier-dominated transition reduces to a
    function of F_saddle alone:

        kappa ~ (D V / h^2) exp(-beta F_saddle)

    Returned without a prefactor fit, so any systematic offset shows up as a
    constant in log space rather than being absorbed silently.
    """
    beta = 1.0 / rt(temperature)
    return (D * cell_area / h ** 2) * np.exp(-beta * np.asarray(F_saddle))


def barrier_features(F, grid, centres, D, temperature=298.0):
    """Per-edge barrier descriptors, for use as PHGNN edge features.

    These are the features that CAN inform kappa, unlike the endpoint free
    energies. All are symmetric in (i, j) by construction, so a model
    consuming them keeps kappa symmetric and detailed balance structural.
    """
    sad = minimax_barrier(F, grid, centres)
    pts = grid.center_grid()
    F = np.asarray(F).ravel()
    n = len(centres)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(sad[i, j]):
                continue
            dv = (pts[centres[j]] - pts[centres[i]] + np.pi) % (2 * np.pi) - np.pi
            rows.append({
                "edge": (i, j),
                "F_saddle": float(sad[i, j]),
                # barrier measured from the higher of the two wells: the
                # quantity that sets the slower direction
                "barrier_from_deeper": float(sad[i, j] - min(F[centres[i]],
                                                             F[centres[j]])),
                "barrier_from_shallower": float(sad[i, j] - max(F[centres[i]],
                                                                F[centres[j]])),
                "separation": float(np.linalg.norm(dv)),
                "log_D": float(np.log10(D)),
            })
    return rows
