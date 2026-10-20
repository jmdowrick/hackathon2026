"""The delta mapping: electrical charge of a state that has no position.

THE PROBLEM
-----------
A Woodhull electrical distance needs a membrane-normal position. A state
discovered by SRV/MSM on permutation-invariant features has none: it is a
region of a latent space, and in the multi-ion case it contains several ions
at once. The tempting fix -- take the mean ion z of each state and read delta
off it -- is wrong three separate ways:

1. MULTI-ION. A state with ions at delta 0.25 and 0.75 holds one full charge
   of displacement; the delta of their mean position is 0.5. A knock-on step
   in which both ions advance one site moves twice the charge of a single-ion
   hop, and a per-state "position" cannot express that.
2. JENSEN. The geometric delta(z) is strongly nonlinear (the filter carries
   most of the drop), so delta(<z>) != <delta(z)> even for one ion.
3. CLOSURE. Any per-state quantity telescopes to ZERO around a closed cycle.
   The net charge a permeation cycle carries lives in the bath species, not in
   the states -- which is why an explicit empty state broke the old telescoping
   (bug 12) and had to be patched by hand.

WHAT IS COMPUTED INSTEAD
------------------------
The quantity the voltage actually couples to. For a state i under applied
voltage V, linear response gives

    -dG_i/dV = e <Q>_i ,     Q(x) = sum_a delta(z_a)

summed over the ions inside the field region, where delta(z) is the
normalised potential profile (0 at the inner face, 1 at the outer). This is
the gating-charge definition, exact at small V. Q is a symmetric sum, so it is
permutation-invariant without any extra construction, and it is a plain
state-conditional average, so weighted-ensemble weights enter as ordinary
weights -- nothing was biased, so nothing needs reweighting.

Per reaction, the charge moved is the change in Q INCLUDING the bath species
the reaction exchanges with, each placed at its face:

    z_r = z_ion * ( Q_b - Q_a + sum_sinks delta_s - sum_sources delta_s )

Around any cycle the state terms telescope away and what is left is the net
number of ions moved from the delta=0 bath to the delta=1 bath. Charge closure
is therefore a consequence of STOICHIOMETRY, not of renormalising the deltas,
and it holds for any Q whatsoever -- including the noisy estimates a finite
trajectory gives.

ION COUNT AND THE WINDOW
------------------------
Ions below the window contribute 0 (delta = 0 there), so the inner boundary is
seamless. Ions above it would contribute 1 each and are instead accounted as
the outer bath species. That bookkeeping is only consistent if the states
RESOLVE how many ions are in the window -- otherwise an ion leaving to the
outer bath changes Q without any reaction in the model carrying the charge.
count_purity reports, per state, the weight on its modal window count; a
state below ~0.9 is mixing occupancies and its Q is not a well-defined
state charge. Per-site occupancy features resolve this; a single-ion CV
does not (see HANDOVER section 4).
"""

from __future__ import annotations

import numpy as np


def delta_profile_fn(z_profile, delta_profile):
    """delta(z) from a sampled profile, clamped to [0, 1] outside it.

    The profile must be normalised 0 -> 1 across its range, which is what
    kcsa.electrical_distance_from_geometry returns.
    """
    zp = np.asarray(z_profile, float)
    dp = np.asarray(delta_profile, float)
    if zp.ndim != 1 or zp.shape != dp.shape or len(zp) < 2:
        raise ValueError("z_profile and delta_profile must be matching 1D arrays")
    order = np.argsort(zp)
    zp, dp = zp[order], dp[order]
    if abs(dp[0]) > 1e-9 or abs(dp[-1] - 1.0) > 1e-9:
        raise ValueError(f"delta profile must run 0 -> 1 across the window, "
                         f"got {dp[0]:.3g} -> {dp[-1]:.3g}")
    if np.any(np.diff(dp) < -1e-12):
        raise ValueError("delta profile must be non-decreasing in z")

    def f(z):
        return np.interp(np.asarray(z, float), zp, dp, left=0.0, right=1.0)

    f.window = (float(zp[0]), float(zp[-1]))
    return f


def frame_charges(z_ions, delta_fn, window=None):
    """Q(x) = sum of delta over the ions inside the window, per frame.

    z_ions : (n_frames, n_ions) membrane-normal ion positions in the SAME frame
             as the delta profile. NaN marks an absent ion (padding).
    Returns (Q, n_in): the charge and the number of ions in the window.
    """
    Z = np.asarray(z_ions, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    lo, hi = window if window is not None else delta_fn.window
    present = np.isfinite(Z)
    inside = present & (Z >= lo) & (Z <= hi)
    d = np.where(inside, delta_fn(np.where(present, Z, lo)), 0.0)
    return d.sum(axis=1), inside.sum(axis=1)


def state_charges(z_ions, labels, n_states, delta_fn, weights=None,
                  window=None):
    """Per-state displacement charge <Q>_i from labelled frames.

    labels  : (n_frames,) metastable-state index per frame; negative = not
              assigned (e.g. outside every core set) and is ignored.
    weights : (n_frames,) frame weights. For weighted-ensemble data these are
              the walker weights; for plain MD leave None.

    Returns Q, its standard error, and diagnostics. The standard error treats
    frames as independent, so it is a LOWER BOUND on the true uncertainty;
    consecutive MD frames are correlated.
    """
    labels = np.asarray(labels, int)
    Qf, n_in = frame_charges(z_ions, delta_fn, window)
    if len(labels) != len(Qf):
        raise ValueError(f"{len(labels)} labels for {len(Qf)} frames")
    w = np.ones(len(Qf)) if weights is None else np.asarray(weights, float)
    if w.shape != Qf.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite, non-negative, one per frame")

    Q = np.full(n_states, np.nan)
    se = np.full(n_states, np.nan)
    n_eff = np.zeros(n_states)
    purity = np.full(n_states, np.nan)
    modal_count = np.full(n_states, -1)
    for s in range(n_states):
        sel = labels == s
        ws = w[sel]
        W = ws.sum()
        if W <= 0:
            continue
        q = Qf[sel]
        Q[s] = float(np.sum(ws * q) / W)
        n_eff[s] = W ** 2 / np.sum(ws ** 2)
        var = float(np.sum(ws * (q - Q[s]) ** 2) / W)
        se[s] = np.sqrt(var / n_eff[s]) if n_eff[s] > 1 else np.nan
        counts = np.bincount(n_in[sel], weights=ws)
        modal_count[s] = int(np.argmax(counts))
        purity[s] = float(counts.max() / W)

    missing = [s for s in range(n_states) if not np.isfinite(Q[s])]
    return {
        "Q": Q, "se_iid_lower_bound": se, "n_eff": n_eff,
        "count_purity": purity, "modal_ion_count": modal_count,
        "unvisited_states": missing,
        "note": ("Q_i = <sum_a delta(z_a)>_i over ions in the field window. "
                 "se treats frames as independent: a lower bound."),
    }


def reaction_charges_from_states(Q, edges, transits=None, species_delta=None,
                                 z_ion=1.0):
    """Per-reaction charge z_r from state charges and bath stoichiometry.

    z_r = z_ion * (Q_b - Q_a + sum_sinks delta - sum_sources delta)

    species_delta maps every species named in a transit to its electrical
    position: the inner bath at 0, the outer at 1. A species that is a
    bookkeeping device rather than a physical ion reservoir (a standard-state
    reference such as K_ref) must be listed EXPLICITLY with None, so that
    leaving a real bath out is an error rather than a silent zero.
    """
    Q = np.asarray(Q, float)
    e = np.asarray(edges, int)
    if not np.all(np.isfinite(Q[np.unique(e)])):
        raise ValueError("a state used by a reaction has no charge estimate "
                         "(never visited?)")
    z = Q[e[:, 1]] - Q[e[:, 0]]
    species_delta = dict(species_delta or {})
    for t in (transits or []):
        for name in (t.source, t.sink):
            if name not in species_delta:
                raise ValueError(
                    f"transit species {name!r} has no electrical position; "
                    "give it one, or None if it carries no physical ion")
        ds = species_delta[t.sink]
        dsrc = species_delta[t.source]
        z[t.reaction] += (0.0 if ds is None else float(ds)) \
            - (0.0 if dsrc is None else float(dsrc))
    return z_ion * z
