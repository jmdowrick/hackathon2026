"""The MSM reference arm (deeptime).

INDEPENDENT BY CONSTRUCTION. The MSM discretises configuration space and counts
transitions; the PHNN learns a continuous field by gradient descent with no
binning at all. They share only the raw trajectories. Their failure modes are
nearly disjoint, so agreement between them is evidence rather than bookkeeping.
That is the project's non-circularity argument, and it is why the MSM is here.

PyEMMA is deprecated (repo archived, points users at deeptime); deeptime is the
maintained successor.

A note worth making on the slide: extracting a continuous-time RATE MATRIX from
a discrete-time T(tau) runs into the embedding problem -- matrix logarithms can
yield negative off-diagonal rates and an embeddable generator need not exist at
all. The PHNN sidesteps this entirely by learning in continuous time. That is a
genuine advantage, correctly stated.
"""

from __future__ import annotations

import numpy as np


def _split(X, traj_id):
    traj_id = np.asarray(traj_id)
    return [np.asarray(X)[traj_id == t] for t in np.unique(traj_id)]


def features(X):
    """sin/cos of the angles -- k-means on raw radians breaks at the +-pi wrap."""
    X = np.asarray(X)
    return np.concatenate([np.sin(X), np.cos(X)], axis=1)


def build(X, traj_id, dt, n_clusters=100, lagtime=10, seed=42,
          lags=(1, 2, 5, 10, 20, 50), n_its=4, n_metastable=4):
    """Cluster, estimate a reversible MSM, and report the standard diagnostics."""
    from deeptime.clustering import KMeans
    from deeptime.markov.msm import MaximumLikelihoodMSM

    trajs = _split(X, traj_id)
    feats = [features(t) for t in trajs]
    allf = np.concatenate(feats)

    cl = KMeans(n_clusters=n_clusters, init_strategy="kmeans++",
                max_iter=500, fixed_seed=seed, n_jobs=1).fit_fetch(allf)
    dtrajs = [np.asarray(cl.transform(f)).astype(int).ravel() for f in feats]

    # implied timescales -- the field's accepted evidence of convergence, and
    # the first thing an expert looks for
    its = {}
    for lag in lags:
        try:
            m = MaximumLikelihoodMSM(reversible=True).fit_fetch(dtrajs, lagtime=int(lag))
            ts = np.asarray(m.timescales())[:n_its] * dt * lag / lag
            its[int(lag)] = (np.asarray(m.timescales())[:n_its] * dt).tolist()
        except Exception as e:                      # pragma: no cover
            its[int(lag)] = [float("nan")] * n_its

    msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(dtrajs, lagtime=int(lagtime))
    ts = np.asarray(msm.timescales()) * dt

    out = {
        "n_clusters": int(n_clusters),
        "lagtime_frames": int(lagtime),
        "lagtime_ps": float(lagtime * dt),
        "timescales_ps": ts[:n_its].tolist(),
        "slowest_ps": float(ts[0]) if ts.size else float("nan"),
        "its_by_lag_ps": its,
        "stationary": np.asarray(msm.stationary_distribution).tolist(),
        "n_states": int(msm.n_states),
    }

    try:
        pcca = msm.pcca(n_metastable)
        assign = np.asarray(pcca.assignments)
        out["pcca_assignments"] = assign.tolist()
        mf = {}
        for a in range(n_metastable):
            for b in range(n_metastable):
                if a == b:
                    continue
                A = np.flatnonzero(assign == a)
                B = np.flatnonzero(assign == b)
                if A.size and B.size:
                    mf[f"{a}->{b}"] = float(msm.mfpt(A, B) * dt)
        out["pcca_mfpt_ps"] = mf
    except Exception as e:                          # pragma: no cover
        out["pcca_error"] = str(e)

    out["cluster_centers_feat"] = np.asarray(cl.cluster_centers).tolist()

    # MFPTs between the SAME geometric basins the bond graph uses.
    #
    # Without this the scorecard compares a bond-graph MFPT for one named basin
    # pair against the median of all PCCA MFPTs -- different quantities, so the
    # comparison is meaningless however large or small the discrepancy. PCCA's
    # metastable sets are discovered by the MSM and need not line up with the
    # geometric basins at all, so we re-derive both from one definition.
    try:
        from .fes import BASINS_DEG
        ang = centers_to_angles(cl.cluster_centers)        # (n_clusters, 2)
        names_b = list(BASINS_DEG)
        ref = np.radians(np.array([BASINS_DEG[k] for k in names_b]))
        d = np.stack([
            np.linalg.norm((ang - c + np.pi) % (2 * np.pi) - np.pi, axis=1)
            for c in ref
        ], axis=1)
        assign = np.argmin(d, axis=1)

        # restrict to the states the MSM actually kept (largest connected set)
        sym = np.asarray(msm.state_symbols()) if hasattr(msm, "state_symbols") \
            else np.arange(msm.n_states)
        pos = {int(s): i for i, s in enumerate(sym)}

        gm = {}
        for a, na in enumerate(names_b):
            for b, nb in enumerate(names_b):
                if a == b:
                    continue
                A = [pos[i] for i in np.flatnonzero(assign == a) if i in pos]
                B = [pos[i] for i in np.flatnonzero(assign == b) if i in pos]
                if A and B:
                    gm[f"{na}->{nb}"] = float(msm.mfpt(np.array(A), np.array(B)) * dt)
        out["basin_mfpt_ps"] = gm
        out["basin_assignment"] = assign.tolist()
        out["basin_names"] = names_b
    except Exception as e:                                  # pragma: no cover
        out["basin_mfpt_error"] = str(e)

    return out, msm, cl, dtrajs


def centers_to_angles(centers_feat):
    """Map sin/cos cluster centres back to (phi, psi)."""
    c = np.asarray(centers_feat)
    return np.stack([np.arctan2(c[:, 0], c[:, 2]), np.arctan2(c[:, 1], c[:, 3])], axis=1)
