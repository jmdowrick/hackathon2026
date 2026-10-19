"""KcsA selectivity filter: landscape provider and D extraction from MD.

THE DIVISION OF LABOUR, AND WHY
-------------------------------
F and D have completely different sampling requirements:

  F needs RARE EVENTS -- barrier crossings. At +300 mV a KcsA permeation study
    sees ~90 conduction events per microsecond; 50 events needs ~0.55 us, and
    every available structure (1BL8, 1K4C, 3EFF) is CLOSED-GATE, giving zero
    permeation unrestrained. Out of reach in a short project.

  D needs only LOCAL DISPLACEMENTS -- ions jiggling in the filter. Reachable in
    tens of nanoseconds, with a closed gate and no permeation whatsoever.

So D is measured here from the user's own MD, and F is taken from the
literature. That split is stated in every artifact this module writes, because
the two halves have very different standing.

PROVENANCE OF THE LANDSCAPE -- READ THIS BEFORE QUOTING A NUMBER
----------------------------------------------------------------
`filter_landscape` is NOT digitised Bernecheey-Roux data. It is a parameterised
surface constrained to reproduce the published, citable features:

  * the largest free-energy barrier is 2-3 kcal/mol (8-13 kJ/mol), from which
    Berneche & Roux concluded conduction is "limited by diffusion"
    (Nature 414:73, 2001, doi 10.1038/35102067)
  * conduction proceeds by transitions between 2-ion and 3-ion filter states
    (Hodgkin-Keynes knock-on), not by single-ion hops
  * wells of comparable depth at the sites (Berneche & Roux, PNAS 2003)

The SITE GEOMETRY is structural fact from the 2.0 A structure 1K4C: four
binding sites S1-S4 formed by successive carbonyl rings of the TVGYG
signature, spaced ~3.3 A, plus S0 above and the cavity site below.

Anything computed from this landscape is therefore "consistent with the
published PMF", not "the published PMF". `provenance()` returns that statement
so it lands in the artifact rather than being remembered.
"""

from __future__ import annotations

import numpy as np

from .thermo.smoluchowski import rt

# --- structural facts (1K4C, 2.0 A) ---------------------------------------
# Site spacing along the pore axis. One site per carbonyl ring of TVGYG.
SITE_SPACING_A = 3.3
# Sites ordered extracellular -> intracellular. z in angstroms, S2 at origin.
SITES = {"S0": 6.6, "S1": 3.3, "S2": 0.0, "S3": -3.3, "S4": -6.6}
# KcsA selectivity filter signature and its residue numbers
FILTER_SEQUENCE = "TVGYG"
FILTER_RESIDUES = (75, 76, 77, 78, 79)       # Thr75 Val76 Gly77 Tyr78 Gly79
CAVITY_Z_A = -13.0                            # water-filled cavity, approx

# --- published kinetics ----------------------------------------------------
PUBLISHED = {
    "max_barrier_kcal": (2.0, 3.0),
    "max_barrier_kJ": (8.4, 12.6),
    "mechanism": "2-ion <-> 3-ion knock-on",
    "conductance_pS_experiment": 100.0,
    "source": "Berneche & Roux, Nature 414:73 (2001); PNAS 100:8644 (2003)",
}


def provenance() -> dict:
    """What came from where. Belongs in every artifact built from this."""
    return {
        "F_source": "literature, parameterised",
        "F_detail": ("Analytic surface constrained to the published maximum "
                     "barrier of 2-3 kcal/mol and the 2-ion/3-ion knock-on "
                     "mechanism. NOT digitised PMF data."),
        "F_citation": PUBLISHED["source"],
        "D_source": "measured from this project's MD",
        "geometry_source": "1K4C (2.0 A): four sites, ~3.3 A spacing, TVGYG",
        "forcefield_caveat": (
            "Fixed-charge force fields underestimate KcsA conductance by "
            "roughly 2-10x (C36m 27.7 pS, C36m-NBFIX 64.0 pS, Amber 51.2 pS "
            "at +300 mV with the gate restrained open, against ~100 pS "
            "experimental). Absolute conductance is therefore indicative; the "
            "barriers and the diffusivity are the outputs with standing."),
    }


# ---------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------

def filter_landscape(z, max_barrier_kJ=10.5, well_depth_kJ=0.0,
                     sites=None, sigma_A=1.1):
    """Free energy of a single K+ along the filter axis, kJ/mol.

    Wells at the crystallographic sites, barriers between them scaled to the
    published maximum. Returned relative to the deepest well.

    This is a ONE-ION reduction of what is really a multi-ion landscape, and
    that is a declared approximation: Berneche & Roux's point is that
    conduction is a 2-ion/3-ion process. Use multi_ion_landscape for the
    cooperative version.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    sites = sites or SITES
    centres = np.array(sorted(sites.values()))

    # A single corrugation whose amplitude IS the site-to-site barrier: zero at
    # every site (cos = 1) and exactly max_barrier_kJ at every midpoint.
    #
    # The earlier version summed Gaussian wells AND a cosine, so the two terms
    # interacted and the realised barrier (14.85 kJ/mol) overshot the published
    # 8.4-12.6 range while the parameter still read 10.5. A parameter that does
    # not equal the thing it is named after is how a landscape quietly drifts
    # outside the range it was supposed to be constrained to.
    k = 2 * np.pi / SITE_SPACING_A
    F = 0.5 * max_barrier_kJ * (1.0 - np.cos(k * (z - centres[0])))

    # Optional per-site depth variation. DEFAULT ZERO, because any non-zero
    # value deepens the sites relative to the midpoints and so raises the
    # realised barrier above the parameter (6.0 kJ/mol of wells added 2.2
    # kJ/mol of barrier, pushing a nominal 10.5 to 12.7 and out of the
    # published range). Berneche & Roux report wells of comparable depth, so
    # equal wells is the faithful default and this is an explicit perturbation.
    if well_depth_kJ:
        for c in centres:
            F -= well_depth_kJ * np.exp(-0.5 * ((z - c) / sigma_A) ** 2)
    return F - F.min()


def realised_barrier(max_barrier_kJ=10.5, **kw) -> float:
    """The barrier the landscape ACTUALLY has, between adjacent sites.

    Reported rather than assumed, and checked against the published range, so
    a parameterisation that drifts out of bounds is caught at build time.
    """
    z = np.linspace(SITES["S4"], SITES["S0"], 2001)
    F = filter_landscape(z, max_barrier_kJ, **kw)
    mid = 0.5 * (SITES["S3"] + SITES["S2"])
    i_mid = int(np.argmin(np.abs(z - mid)))
    i_site = int(np.argmin(np.abs(z - SITES["S2"])))
    return float(F[i_mid] - F[i_site])


def check_against_published(max_barrier_kJ=10.5, **kw) -> dict:
    lo, hi = PUBLISHED["max_barrier_kJ"]
    b = realised_barrier(max_barrier_kJ, **kw)
    return {"realised_barrier_kJ": b, "published_range_kJ": [lo, hi],
            "in_range": bool(lo <= b <= hi),
            "citation": PUBLISHED["source"]}


def multi_ion_landscape(z1, z2, max_barrier_kJ=10.5, repulsion_kJ=12.0,
                        min_sep_A=3.0):
    """Two-ion landscape: single-ion terms plus electrostatic repulsion.

    The knock-on mechanism lives in the repulsion term -- two ions cannot
    occupy adjacent sites comfortably, so the favourable path is concerted.
    A one-ion landscape cannot express that, which is why the published
    mechanism is a multi-ion one.
    """
    z1 = np.asarray(z1, float); z2 = np.asarray(z2, float)
    F = filter_landscape(z1, max_barrier_kJ) + filter_landscape(z2, max_barrier_kJ)
    d = np.abs(z1 - z2)
    F = F + repulsion_kJ * np.exp(-(d / min_sep_A) ** 2)
    return F - np.nanmin(F)


def site_free_energies(max_barrier_kJ=10.5, **kw):
    """Free energy at each crystallographic site."""
    names = sorted(SITES, key=lambda s: SITES[s])
    z = np.array([SITES[n] for n in names])
    F = filter_landscape(z, max_barrier_kJ, **kw)
    return dict(zip(names, F - F.min()))


# ---------------------------------------------------------------------------
# D from the user's own MD
# ---------------------------------------------------------------------------

def filter_axis_from_positions(filter_ca_xyz):
    """Pore axis and origin from the four filter Thr Calpha atoms.

    The axis is the C4 symmetry axis; the origin is the centroid. Both are
    taken from the structure rather than assumed to be the box z-axis, because
    the protein drifts and tilts during equilibration and an assumed axis
    quietly projects the ion coordinate onto the wrong direction.
    """
    P = np.asarray(filter_ca_xyz, dtype=float)
    if P.ndim != 2 or P.shape[0] < 3:
        raise ValueError("need at least 3 filter atoms")
    origin = P.mean(axis=0)
    # the pore axis is the direction of least variance of the ring
    u, s, vt = np.linalg.svd(P - origin)
    axis = vt[-1]
    if axis[2] < 0:
        axis = -axis                  # orient towards +z (extracellular)
    return origin, axis / np.linalg.norm(axis)


def project_ions(ion_xyz, origin, axis):
    """Ion positions along the pore axis, relative to the filter origin.

    ion_xyz : (n_frames, n_ions, 3)
    returns (n_frames, n_ions) coordinate along the axis
    """
    X = np.asarray(ion_xyz, dtype=float)
    return np.einsum("fij,j->fi", X - np.asarray(origin), np.asarray(axis))


def measure_filter_D(z_traj, dt, z_min=-9.0, z_max=9.0, lag=1,
                     n_bins=12, min_count=50):
    """D(z) for K+ inside the filter, from the second Kramers-Moyal moment.

    z_traj : (n_frames, n_ions) ion coordinate along the pore axis, angstroms
    dt     : frame spacing in ps
    returns D in A^2/ps per bin, plus counts and the global value

    IONS ENTERING AND LEAVING ARE HANDLED EXPLICITLY. Only displacement pairs
    where the SAME ion is inside the filter window at both times contribute; a
    pair spanning an entry or exit is a jump between different physical
    situations, not a diffusive step, and including it inflates D. This is the
    permeation analogue of the trajectory-boundary check used elsewhere in the
    package.

    No periodic wrapping is applied: the filter window is small and interior,
    so an ion crossing the box boundary is outside the window by construction
    and is already excluded.
    """
    Z = np.asarray(z_traj, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n_f, n_i = Z.shape
    if n_f <= lag:
        raise ValueError("trajectory shorter than the lag")

    z0 = Z[:-lag]
    z1 = Z[lag:]
    inside = (z0 >= z_min) & (z0 <= z_max) & (z1 >= z_min) & (z1 <= z_max)
    if not inside.any():
        return {"available": False, "note": "no ion-frames inside the filter"}

    dz = (z1 - z0)[inside]
    zc = z0[inside]
    tau = dt * lag

    edges = np.linspace(z_min, z_max, n_bins + 1)
    idx = np.clip(np.digitize(zc, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    s2 = np.bincount(idx, weights=dz ** 2, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = s2 / (2.0 * tau * counts)

    glob = float(np.mean(dz ** 2) / (2.0 * tau))
    thin = counts < min_count
    D = np.where(thin, glob, D)
    D = np.where(np.isfinite(D), D, glob)

    return {
        "available": True,
        "D_per_bin": D, "bin_centres": 0.5 * (edges[:-1] + edges[1:]),
        "counts": counts, "n_thin_bins": int(thin.sum()),
        "D_global": glob, "lag": lag, "tau_ps": tau,
        "n_pairs": int(inside.sum()),
        "units": "angstrom^2/ps",
    }


def D_lag_stability(z_traj, dt, lags=(1, 2, 5, 10), **kw):
    """Is the filter D lag-independent?

    Confined motion in a site makes apparent D fall with lag, exactly as the
    dihedral case did. A strong lag dependence means the ion coordinate alone
    is not Markovian in the filter, which is worth reporting rather than
    tuning away.
    """
    out = {}
    for L in lags:
        try:
            r = measure_filter_D(z_traj, dt, lag=int(L), **kw)
            out[int(L)] = r["D_global"] if r.get("available") else float("nan")
        except Exception:
            out[int(L)] = float("nan")
    v = np.array([x for x in out.values() if np.isfinite(x) and x > 0])
    ratio = float(v.max() / v.min()) if v.size > 1 else float("nan")
    return {"D_by_lag": out, "max_ratio": ratio,
            "markovian": bool(np.isfinite(ratio) and ratio < 2.0)}


def load_from_mdtraj(traj, filter_resids=FILTER_RESIDUES, ion_name="K"):
    """Extract filter-axis K+ coordinates from an mdtraj Trajectory.

    Returns z in ANGSTROMS (mdtraj works in nm) along the pore axis, with the
    axis recomputed per frame so protein tilt does not leak into the ion
    coordinate.
    """
    import mdtraj as md

    top = traj.topology
    ions = top.select(f"name {ion_name} or resname {ion_name}")
    if len(ions) == 0:
        raise ValueError(f"no ions matched name/resname {ion_name!r}")
    sel = " or ".join(f"resSeq {r}" for r in filter_resids)
    ca = top.select(f"name CA and ({sel})")
    if len(ca) < 3:
        raise ValueError(f"found only {len(ca)} filter CA atoms")

    z = np.empty((traj.n_frames, len(ions)))
    for f in range(traj.n_frames):
        origin, axis = filter_axis_from_positions(traj.xyz[f, ca])
        z[f] = (traj.xyz[f, ions] - origin) @ axis
    return z * 10.0, {"n_ions": int(len(ions)), "n_filter_ca": int(len(ca)),
                      "dt_ps": float(traj.timestep) if traj.timestep else None}


# ---------------------------------------------------------------------------
# Assemble the permeation bond graph
# ---------------------------------------------------------------------------

def permeation_graph(D_filter, max_barrier_kJ=10.5, temperature=298.0,
                     c_in=140.0, c_out=4.0, sites=None, F_profile=None,
                     z_profile=None):
    """Site states -> electrogenic bond graph with an I-V curve.

    RATES COME FROM THE BARRIER, NOT THE ENDPOINT FREE ENERGIES.

        k_ij = (D / h^2) exp(-beta (F_dagger_ij - F_i))

    An earlier version used the Bicout-Szabo adjacent-bin form, which depends
    only on F_j - F_i. Between site states separated by a resolved barrier
    that form ignores the barrier completely: with the filter's equal-depth
    sites it made every kappa identical and the landscape had NO influence on
    the kinetics whatsoever. The symptom was every K_i equal to 1. See
    barrier.kramers_rate.

    F_profile / z_profile supply a landscape sampled along the pore axis. Pass
    a LEARNED one (see permeation_graph_from_trajectory) to drive the model
    from trajectory data rather than from the analytic stand-in.
    """
    from .barrier import kramers_rate, minimax_1d
    from .electro import (ElectroBondGraph, Species, Transit,
                          electrical_distances, reaction_charges)

    sites = sites or SITES
    names = sorted(sites, key=lambda s: sites[s])      # intracellular -> out
    zpos = np.array([sites[n] for n in names])

    if F_profile is None:
        z_profile = np.linspace(zpos.min(), zpos.max(), 401)
        F_profile = filter_landscape(z_profile, max_barrier_kJ)
        landscape_source = "analytic stand-in (literature-constrained)"
    else:
        F_profile = np.asarray(F_profile, float)
        z_profile = np.asarray(z_profile, float)
        landscape_source = "supplied (e.g. learned from trajectory)"

    # site free energies and the saddles between them, read off the profile
    site_idx = [int(np.argmin(np.abs(z_profile - zz))) for zz in zpos]
    F = F_profile[site_idx] - np.nanmin(F_profile[site_idx])
    saddles = minimax_1d(F_profile, z_profile, site_idx)

    n = len(names)
    edges = np.array([[i, (i + 1) % n] for i in range(n)])
    h = SITE_SPACING_A
    D = float(D_filter)

    kf, kr = [], []
    for e, (i, j) in enumerate(edges):
        if e < len(saddles):
            Fs = saddles[e]
        else:
            # the closing transit through the bath: take the largest barrier
            # as a stand-in for the access/exit step
            Fs = float(np.nanmax(saddles)) if len(saddles) else F.max()
        # A^2/ps -> 1/s
        kf.append(float(kramers_rate(Fs, F[i], D, h, temperature)) * 1e12)
        kr.append(float(kramers_rate(Fs, F[j], D, h, temperature)) * 1e12)

    delta = electrical_distances(zpos, zpos.min(), zpos.max())
    z = reaction_charges(delta, edges, z_ion=1.0)

    species = {"K_in": Species("K_in", 1.0, float(c_in)),
               "K_out": Species("K_out", 1.0, float(c_out))}
    transits = [Transit(reaction=len(edges) - 1, source="K_in", sink="K_out")]

    bg = ElectroBondGraph.from_rates(np.array(kf), np.array(kr), edges, z,
                                     F_states=F, temperature=temperature,
                                     labels=names, mode="single_channel",
                                     species=species, transits=transits)
    bg.meta.update({"system": "KcsA selectivity filter",
                    "sites": names, "site_z_A": zpos.tolist(),
                    "site_F_kJ": F.tolist(),
                    "saddles_kJ": [float(x) for x in saddles],
                    "D_used_A2_per_ps": D,
                    "rate_law": "Kramers over the resolved barrier",
                    "landscape_source": landscape_source,
                    **provenance()})
    return bg


def learn_landscape(z_traj, dt, n_grid=160, z_min=-9.0, z_max=9.0,
                    traj_id=None, steps=3000, min_count=30, temperature=298.0,
                    seed=0, verbose=False, n_per_dim=120, n_harmonics=None,
                    width=32, depth=0):
    """Learn F(z) for a filter ion from trajectory data, via the PHNN.

    This is what makes the model DERIVED rather than asserted: D is measured
    from the second Kramers-Moyal moment and frozen, H is the only learned
    object, and both come from the same trajectories and nothing else.

    Uses the SPARSE one-dimensional path (nd.train_nd), not a dense 2D grid.
    The first version padded the 1D coordinate into a 2D TorusGrid with a
    dummy second axis; that put every sample in 160 of 1280 bins, spent the
    network's capacity on an axis with no signal, and overfitted immediately
    (r2 = -0.37, early stop at step 100, a perfectly flat landscape). The
    sparse path touches only occupied voxels and has no dummy axis.

    Segments are split wherever an ion LEAVES the window. Filtering
    out-of-window frames and concatenating what remains silently closes the
    gap, turning an excursion into one enormous displacement -- the same bug
    class as the trajectory-boundary check elsewhere in this package.
    """
    from . import nd
    from .periodic import MixedCVSpec

    Z = np.asarray(z_traj, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n_f, n_i = Z.shape

    period = float(z_max - z_min)
    spec = MixedCVSpec(names=("z",), periods=(period,), origins=(z_min,),
                       units=("angstrom",))

    segs, ids = [], []
    for i in range(n_i):
        col = Z[:, i]
        inside = (col >= z_min) & (col <= z_max)
        # contiguous runs only
        d = np.diff(inside.astype(int))
        starts = list(np.flatnonzero(d == 1) + 1) + ([0] if inside[0] else [])
        ends = list(np.flatnonzero(d == -1) + 1) + ([len(col)] if inside[-1] else [])
        for a_, b_ in zip(sorted(starts), sorted(ends)):
            if b_ - a_ >= 50:
                segs.append(col[a_:b_])
                ids.append(np.full(b_ - a_, len(ids)))
    if not segs:
        raise ValueError("no ion stayed inside the filter window long enough")

    zz = np.concatenate(segs)
    tid = np.concatenate(ids)
    TH = spec.to_angles(zz[:, None])          # (n, 1) on the torus

    D_ang = diffusion_in_angle_units(zz, dt, tid, spec)
    # Enough harmonics to express the site periodicity across the window,
    # with headroom. period/spacing = how many oscillations must be resolved.
    if n_harmonics is None:
        n_harmonics = int(np.ceil(period / SITE_SPACING_A)) + 3

    # depth=0 gives a LINEAR readout on the Fourier features, i.e. H is a
    # Fourier series -- exactly the right model class for a periodic 1D
    # landscape, with ~2*n_harmonics parameters instead of thousands.
    #
    # The deep default (width 64, depth 2) puts ~5k parameters on ~96 training
    # voxels: it overfits so immediately that early stopping fires at step 0,
    # the held-out r2 goes NEGATIVE, and the returned landscape is nearly
    # flat. The structural guarantees are untouched by the smaller model --
    # the drift is still -D grad H with H a scalar potential, so the second
    # law and detailed balance still hold by construction.

    model, hist, info = nd.train_nd(TH, dt, D_ang, dim=1, n_per_dim=n_per_dim,
                                    temperature=temperature, traj_id=tid,
                                    steps=steps, min_count=min_count,
                                    seed=seed, verbose=verbose,
                                    n_harmonics=n_harmonics,
                                    width=width, depth=depth)
    info["n_harmonics"] = int(n_harmonics)

    import torch
    z_out = np.linspace(z_min, z_max, n_grid, endpoint=False)
    th_out = spec.to_angles(z_out[:, None]).astype(np.float32)
    with torch.no_grad():
        F = model(torch.tensor(th_out)).numpy().astype(float)
    return {"z": z_out, "F": F - F.min(), "fit": info, "D_angle": D_ang,
            "spec": spec, "n_segments": len(segs)}


def diffusion_in_angle_units(z, dt, traj_id, spec):
    """D in angle^2/ps, for the torus-based estimators."""
    from .cvs import wrapped_delta
    th = spec.to_angles(np.asarray(z)[:, None])[:, 0]
    tid = np.asarray(traj_id)
    ok = tid[:-1] == tid[1:]
    d = wrapped_delta(th[1:][ok], th[:-1][ok])
    return float(np.mean(d ** 2) / (2.0 * dt))


def permeation_graph_from_trajectory(z_traj, dt, traj_id=None, **kw):
    """End-to-end: trajectory -> measured D -> learned F -> bond graph.

    Nothing analytic enters the landscape. If the trajectory changes, the
    model changes; if the pipeline were quietly falling back on the analytic
    stand-in, the result would be independent of the data, which
    test_model_tracks_the_trajectory checks directly.
    """
    learned = learn_landscape(z_traj, dt, traj_id=traj_id,
                              **{k: v for k, v in kw.items()
                                 if k in ("n_grid", "z_min", "z_max", "steps",
                                          "min_count", "seed", "verbose")})
    Dm = measure_filter_D(z_traj, dt)
    if not Dm.get("available"):
        raise ValueError("could not measure D inside the filter")

    bg = permeation_graph(
        Dm["D_global"],
        F_profile=learned["F"], z_profile=learned["z"],
        **{k: v for k, v in kw.items()
           if k in ("temperature", "c_in", "c_out", "sites")})
    bg.meta["landscape_source"] = "LEARNED from trajectory (PHNN)"
    bg.meta["D_source"] = "measured from this trajectory"
    bg.meta["phnn_fit"] = {k: v for k, v in learned["fit"].items()
                           if k in ("r2", "best_step", "n_bins_fit")}
    return bg, learned


def states_from_trajectory(z_traj, dt, traj_id=None, z_min=-9.0, z_max=9.0,
                           n_clusters=80, lagtime=20, max_states=8,
                           verbose=False):
    """Let the DYNAMICS choose the filter states, then check crystallography.

    The crystallographic sites are an input we do not have to make. Building
    an MSM and reading the state count off the implied-timescale spectral gap
    measures it instead, and comparing the recovered centres with S0-S4 is
    then a VALIDATION rather than an assumption.

    On synthetic filter trajectories this recovers the five sites to ~0.02 A.
    It also reports when the gap is weak: equal-depth sites give poor
    metastability, so a low-confidence verdict is the honest output and not a
    failure. That is the same instrument the SGLT1 question needs -- does the
    metastable state count converge on the number the kinetic scheme assumes?
    """
    from . import minimal
    from .periodic import MixedCVSpec

    Z = np.asarray(z_traj, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n_f, n_i = Z.shape
    if traj_id is None:
        zz = Z.T.ravel()
        tid = np.repeat(np.arange(n_i), n_f)
    else:
        zz, tid = np.asarray(z_traj).ravel(), np.asarray(traj_id).ravel()

    spec = MixedCVSpec(names=("z",), periods=(float(z_max - z_min),),
                       origins=(float(z_min),), units=("angstrom",))
    TH = spec.to_angles(zz[:, None])
    TH2 = np.concatenate([TH, np.zeros_like(TH)], axis=1)

    m = minimal.minimal_states(TH2, tid, dt, n_clusters=n_clusters,
                               lagtime=lagtime, max_states=max_states,
                               verbose=verbose)
    centres = np.sort(spec.from_angles(m["centers"])[:, 0])
    inside = centres[(centres >= z_min + 0.5) & (centres <= z_max - 0.5)]

    xtal = np.array(sorted(SITES.values()))
    matched = []
    for c in xtal:
        if inside.size:
            j = int(np.argmin(np.abs(inside - c)))
            matched.append(float(inside[j] - c))
        else:
            matched.append(float("nan"))
    dev = np.array(matched)

    return {
        "n_states_measured": int(m["n_states"]),
        "n_sites_crystal": int(len(xtal)),
        "centres_A": centres.tolist(),
        "centres_in_window_A": inside.tolist(),
        "crystal_sites_A": xtal.tolist(),
        "deviation_A": dev.tolist(),
        "max_deviation_A": float(np.nanmax(np.abs(dev))) if dev.size else float("nan"),
        "agrees_with_crystallography": bool(
            np.nanmax(np.abs(dev)) < 1.0 if dev.size else False),
        "gap": m["gap"],
        "timescales_ps": m["timescales_ps"],
        "note": ("State count and positions MEASURED from the dynamics. "
                 "Crystallography is the cross-check, not the input. A "
                 "low-confidence spectral gap means the sites are not "
                 "strongly metastable, which for equal-depth wells is "
                 "expected and is reported rather than overridden."),
    }
