"""Minimal discretisation: let the DYNAMICS choose the states, not a grid.

THE PROBLEM THIS SOLVES
-----------------------
Every grid-based component in this package costs n^d -- the histogram FES, the
grid selection, the Bicout-Szabo generator, the committor solve, and hence the
bond graph itself. At 2 CVs that is 3,600 states; at 4-6 CVs it is gone. And
the grid is chosen by a scan over candidate resolutions, which is an argument,
not a measurement.

A free-energy landscape discretised without constraint also produces far more
states than the dynamics actually supports: 1,296 Ce elements for a molecule
with four metastable conformers is not a model, it is a picture of a histogram.

THE FIX
-------
Build a Markov state model, read the number of metastable states off the
IMPLIED-TIMESCALE SPECTRAL GAP, and coarse-grain to exactly that many with
PCCA+. The state count becomes a measured quantity with an uncertainty, not a
modelling choice -- and the bond graph that comes out has as many states as the
dynamics supports and no more.

Cost scales with DATA, not with dimension or with n^d. The same code works for
2 dihedrals or for twenty CVs of a transporter.

WHAT THE GAP MEANS
------------------
If timescales t_1 >> t_2 >> ... and there is a clear drop after t_{k-1}, then k
metastable states capture every process slower than the gap, and everything
faster is equilibrated within a state. A WEAK gap is informative too: it says
the system does not have a clean k-state description, which is exactly the
question to answer empirically before asserting a six-state kinetic scheme.
"""

from __future__ import annotations

import numpy as np


def spectral_gap(timescales, max_states=10):
    """Rank candidate state counts by the implied-timescale gap.

    For k states there are k-1 slow processes, so the relevant gap is
    t_{k-1} / t_k. Returns the ranked candidates and the full gap profile so
    the choice can be shown rather than asserted.
    """
    ts = np.asarray([t for t in timescales if np.isfinite(t) and t > 0])
    if ts.size < 2:
        return {"recommended": 2, "gaps": {}, "timescales": ts.tolist(),
                "confident": False,
                "note": "fewer than two resolved timescales; cannot measure a gap"}

    gaps = {}
    for k in range(2, min(max_states, ts.size) + 1):
        # k states -> k-1 slow processes -> gap between process k-1 and k
        if k - 1 < ts.size:
            num = ts[k - 2]
            den = ts[k - 1] if k - 1 < ts.size else None
            if den and den > 0:
                gaps[k] = float(num / den)

    if not gaps:
        return {"recommended": 2, "gaps": {}, "timescales": ts.tolist(),
                "confident": False, "note": "no gap computable"}

    best = max(gaps, key=gaps.get)
    ratio = gaps[best]
    runner = sorted(gaps.values(), reverse=True)
    second = runner[1] if len(runner) > 1 else 1.0
    return {
        "recommended": int(best),
        "gap_ratio": float(ratio),
        "gaps": {int(k): float(v) for k, v in gaps.items()},
        "timescales": ts.tolist(),
        # a gap is only meaningful if it stands out from the others
        "confident": bool(ratio > 2.0 and ratio > 1.5 * second),
        "note": ("clear spectral gap" if ratio > 2.0 and ratio > 1.5 * second
                 else "NO CLEAR GAP: the data does not support a clean "
                      "k-state description. Report this rather than forcing k."),
    }


def minimal_states(X, traj_id, dt, n_clusters=100, lagtime=10, seed=42,
                   n_states=None, max_states=8, verbose=True):
    """MSM -> spectral gap -> PCCA+ -> a minimal set of metastable states.

    Returns the coarse state assignment of every frame, the state centres in
    CV space, and the evidence for the chosen state count.
    """
    from deeptime.clustering import KMeans
    from deeptime.markov.msm import MaximumLikelihoodMSM

    log = print if verbose else (lambda *a, **k: None)
    X = np.asarray(X)
    traj_id = np.asarray(traj_id)
    uniq = np.unique(traj_id)
    trajs = [X[traj_id == t] for t in uniq]

    def feat(a):
        return np.concatenate([np.sin(a), np.cos(a)], axis=1)

    feats = [feat(t) for t in trajs]
    cl = KMeans(n_clusters=n_clusters, init_strategy="kmeans++", max_iter=500,
                fixed_seed=seed, n_jobs=1).fit_fetch(np.concatenate(feats))
    dtrajs = [np.asarray(cl.transform(f)).astype(int).ravel() for f in feats]

    msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(dtrajs, lagtime=int(lagtime))
    ts = np.asarray(msm.timescales()) * dt

    gap = spectral_gap(ts, max_states=max_states)
    k = int(n_states or gap["recommended"])
    log(f"     implied timescales (ps): "
        + ", ".join(f"{t:.1f}" for t in ts[:max_states]))
    log(f"     spectral gap -> {gap['recommended']} states "
        f"(ratio {gap.get('gap_ratio', float('nan')):.2f}); {gap['note']}")
    if n_states:
        log(f"     using k={k} (caller override)")

    pcca = msm.pcca(k)
    micro_to_macro = np.asarray(pcca.assignments)

    # map every frame to a macrostate, via its microstate
    sym = (np.asarray(msm.state_symbols())
           if hasattr(msm, "state_symbols") else np.arange(msm.n_states))
    full = -np.ones(n_clusters, dtype=int)
    for pos, s in enumerate(sym):
        full[int(s)] = micro_to_macro[pos]

    labels = np.concatenate([full[d] for d in dtrajs])
    order = np.concatenate([np.flatnonzero(traj_id == t) for t in uniq])
    macro = -np.ones(len(X), dtype=int)
    macro[order] = labels

    # state centres: circular mean of the member frames
    centers = np.zeros((k, X.shape[1]))
    pops = np.zeros(k)
    for s in range(k):
        sel = macro == s
        pops[s] = sel.sum()
        if sel.any():
            centers[s] = np.arctan2(np.sin(X[sel]).mean(axis=0),
                                    np.cos(X[sel]).mean(axis=0))

    log(f"     macrostate populations: {(pops / max(pops.sum(), 1)).round(3)}")
    return {
        "labels": macro, "centers": centers, "populations": pops,
        "n_states": k, "gap": gap, "lagtime_ps": float(lagtime * dt),
        "timescales_ps": ts.tolist(), "n_microstates": int(n_clusters),
    }


def coarse_rate_matrix(labels, traj_id, lag, dt, n_states):
    """Reversible rate matrix on the macrostates, from transition counts.

    Detailed balance is imposed by symmetrising the count matrix, which is
    precisely the Wegscheider condition the bond graph needs -- so the kappa
    that comes out is symmetric by construction, exactly as in the grid route,
    but with no grid.
    """
    labels = np.asarray(labels)
    traj_id = np.asarray(traj_id)
    ok = (traj_id[:-lag] == traj_id[lag:]) & (labels[:-lag] >= 0) & (labels[lag:] >= 0)
    a, b = labels[:-lag][ok], labels[lag:][ok]

    C = np.zeros((n_states, n_states))
    np.add.at(C, (a, b), 1.0)
    Csym = 0.5 * (C + C.T)

    n_i = Csym.sum(axis=1)
    pi = n_i / max(n_i.sum(), 1.0)
    tau = lag * dt

    K = np.zeros((n_states, n_states))
    nz = n_i > 0
    K[nz] = Csym[nz] / (n_i[nz, None] * tau)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(K, -K.sum(axis=1))
    return K, pi, Csym


def lag_robustness(labels, traj_id, dt, n_states, lags=(1, 2, 5, 10, 20, 50)):
    """Is the coarse rate matrix lag-insensitive?

    A genuinely Markovian coarse-graining gives rates that do not drift with
    lag. Drift means the macrostates are not metastable enough -- the same
    diagnostic logic as the D-vs-lag memory test, applied to the discretisation
    instead of the diffusivity.
    """
    out = {}
    for L in lags:
        try:
            K, pi, _ = coarse_rate_matrix(labels, traj_id, int(L), dt, n_states)
            off = K[~np.eye(n_states, dtype=bool)]
            off = off[off > 0]
            out[int(L)] = float(np.median(off)) if off.size else float("nan")
        except Exception:
            out[int(L)] = float("nan")
    vals = np.array([v for v in out.values() if np.isfinite(v) and v > 0])
    ratio = float(vals.max() / vals.min()) if vals.size > 1 else float("nan")
    return {"median_rate_by_lag": out, "max_ratio": ratio,
            "markovian": bool(np.isfinite(ratio) and ratio < 2.0)}
