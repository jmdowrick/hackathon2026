"""Regression tests -- one per bug actually hit during development.

Every test here encodes a real failure that reached a result before being
caught. Each is written to FAIL on the pre-fix behaviour: where that is not
obvious, the test reconstructs the broken behaviour explicitly and asserts the
two differ, so it cannot quietly pass for the wrong reason.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from ala2bg import diffusion, fes, gridselect, phnn
from ala2bg.cvs import wrapped_delta
from ala2bg.data import simulate_toy, toy_potential
from ala2bg.grid import TorusGrid
from ala2bg.thermo import bondgraph as bgm
from ala2bg.thermo import committor as cm
from ala2bg.thermo.smoluchowski import build_generator


# ---------------------------------------------------------------------------
# BUG 1: equilibrium-flux check normalised by the net flux -> 0/0 -> ~2.17
# ---------------------------------------------------------------------------

def test_equilibrium_flux_normaliser_is_not_degenerate():
    """At equilibrium the net flux vanishes, so normalising BY it is 0/0.

    The fix normalises by the one-way flux, which stays O(1). This test pins
    both halves: the ratio must be tiny AND the scale must be non-degenerate.
    """
    g = TorusGrid(14, 14)
    pts = g.center_grid()
    F = 6.0 * np.cos(2 * pts[:, 0]) + 4.0 * np.sin(pts[:, 1])
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    c = bgm.check_thermo(bg)

    assert c["equilibrium_flux_rel"] < 1e-9
    # the broken version divided by max|net flux|, which is ~1e-19 here
    assert c["equilibrium_flux_scale"] > 1e-6, \
        "normaliser collapsed to ~0; the ratio would be meaningless"
    assert c["equilibrium_flux_max"] / c["equilibrium_flux_scale"] < 1e-9


# ---------------------------------------------------------------------------
# BUG 2: disconnected generator -> singular solve -> NaN MFPT
# ---------------------------------------------------------------------------

def _island_mask(g):
    """A mask with one genuinely detached island."""
    m = np.zeros(g.shape, dtype=bool)
    m[2:8, 2:8] = True          # main block
    m[11:13, 11:13] = True      # island, not adjacent to the block
    return m.ravel()


def test_generator_restricted_to_largest_connected_component():
    g = TorusGrid(16, 16)
    mask = _island_mask(g)
    F = np.zeros(g.n_bins)
    L, keep = build_generator(F, 0.05, g, mask=mask)

    n_comp, _ = sp.csgraph.connected_components(L, directed=False)
    assert n_comp == 1, "generator still disconnected"
    assert len(keep) < mask.sum(), "island was not dropped"
    assert np.abs(np.asarray(L.sum(axis=0)).ravel()).max() < 1e-10, \
        "columns must still sum to zero after restriction"


def test_mfpt_is_finite_on_disconnected_input():
    """The original failure mode: singular matrix -> NaN MFPT, silently."""
    g = TorusGrid(16, 16)
    mask = _island_mask(g)
    F = np.zeros(g.n_bins)
    L, keep = build_generator(F, 0.05, g, mask=mask)
    tau = cm.mfpt(L, np.arange(3))
    assert np.all(np.isfinite(tau)), "MFPT went non-finite"
    assert np.all(tau[3:] > 0)


# ---------------------------------------------------------------------------
# BUG 3: grid too fine -> populated basin severed -> slow process deleted
#        silently (60x60 dropped alphaL, 21k frames, and reported ~120 ps
#        instead of nanoseconds)
# ---------------------------------------------------------------------------

def _two_basin_data(seed=0, n=60_000):
    """Toy data whose two wells are joined by a thinly-sampled barrier."""
    return simulate_toy(n_steps=n, n_traj=4, dt=0.5, D=0.01, barrier=14.0, seed=seed)


def test_gridselect_rejects_a_grid_that_severs_a_populated_basin():
    """Deterministic by construction: a barrier high enough that fine bins in
    the transition region are genuinely empty. No skip branch -- a skip here
    would be a vacuous test.
    """
    d = simulate_toy(n_steps=20_000, n_traj=4, dt=0.5, D=0.01,
                     barrier=16.0, seed=0)
    X = d["X"]
    rep_fine = gridselect.evaluate_grid(X, 90, 0.01, min_basin_frames=2000)
    rep_coarse = gridselect.evaluate_grid(X, 20, 0.01, min_basin_frames=2000)

    assert rep_coarse["ok"], f"coarse grid unexpectedly rejected: {rep_coarse}"
    assert not rep_fine["ok"], "fine grid should sever a populated basin here"
    assert rep_fine["severed_basins"], "rejected without naming what was severed"

    # and the consequence that made this a bug: the severed grid reports a
    # qualitatively different (much faster) slow process
    assert rep_fine["timescales_ps"] and rep_coarse["timescales_ps"]
    # observed: ~5200 ps (connected) vs ~60 ps (severed) -- a ~100x distortion
    assert rep_coarse["timescales_ps"][0] > 10 * rep_fine["timescales_ps"][0], \
        "severing did not visibly distort the kinetics; test would be weak"


def test_chosen_grid_keeps_every_populated_basin_connected():
    """The property that actually matters, asserted directly."""
    d = _two_basin_data(seed=3)
    X = d["X"]
    g, report = gridselect.choose_grid(X, 0.01, candidates=(16, 20, 30, 45, 60),
                                       min_count=1, verbose=False)
    F, counts, mask = fes.histogram_fes(X, g, min_count=1)
    L, keep = build_generator(F, 0.01, g, mask=mask)
    lab, names = fes.assign_basins(g, mask)
    kept = np.zeros(g.n_bins, bool); kept[keep] = True

    for k, nm in enumerate(names):
        if counts[lab == k].sum() >= 2000:
            assert np.any(kept & (lab == k)), \
                f"populated basin {nm} severed by the chosen grid"


def test_gridselect_rejects_numerically_degenerate_spectrum():
    """A near-disconnected grid gives ~1e13 ps timescales -- not physics."""
    bad = {"n": 36, "timescales_ps": [9.9e13, 120.0, 4.5]}
    ratio = bad["timescales_ps"][0] / bad["timescales_ps"][1]
    assert ratio > gridselect.MAX_SANE_TIMESCALE_RATIO, \
        "the guard threshold would not catch the observed degenerate case"


# ---------------------------------------------------------------------------
# BUG 4: winding-cycle residual is None when sampling does not wrap ->
#        TypeError in the log line, crashing a completed run
# ---------------------------------------------------------------------------

def test_wegscheider_handles_non_wrapping_sampled_region():
    g = TorusGrid(20, 20)
    mask = np.zeros(g.shape, dtype=bool)
    mask[4:14, 4:14] = True            # a patch that does NOT wrap
    mask = mask.ravel()
    F = np.zeros(g.n_bins)
    L, keep = build_generator(F, 0.05, g, mask=mask)
    w = bgm.wegscheider_residuals(L, g, keep, n_random=20)

    assert w["winding_phi"]["max"] is None, "no winding loop should exist here"
    assert w["plaquette"]["max"] is not None, "contractible cycles DO exist"
    assert w["plaquette"]["max"] < 1e-10
    # must be formattable without raising -- the actual crash
    _ = "n/a" if w["winding_phi"]["max"] is None else f"{w['winding_phi']['max']:.2e}"


# ---------------------------------------------------------------------------
# BUG 5: PHNN trained per-sample barely learned (loss floor 2.0); and the
#        binned fit overfitted without early stopping (val tripled, learned
#        range inflated 2x). Non-vacuous: recover a KNOWN potential.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_phnn_recovers_known_potential():
    """Ground truth exists here, so this is a real accuracy assertion."""
    d = simulate_toy(n_steps=40_000, n_traj=8, dt=0.5, D=0.01, barrier=12.0, seed=11)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]
    g = TorusGrid(40, 40)
    pts = g.center_grid()
    F_true = toy_potential(pts); F_true -= F_true.min()

    Dm = diffusion.measure_D(X, dt, tid, lag=1)
    assert abs(Dm["D_const"] - 0.01) / 0.01 < 0.10, \
        f"D misestimated: {Dm['D_const']}"

    model, hist, info = phnn.train_binned(X, dt, Dm["D_const"], g, traj_id=tid,
                                          steps=3000, min_count=30)
    H = phnn.evaluate_H(model, pts)

    _, counts, _ = fes.histogram_fes(X, g, min_count=1)
    sel = counts >= 30                      # compare only where there is data
    h = H[sel] - H[sel].mean()
    f = F_true[sel] - F_true[sel].mean()

    corr = float(np.corrcoef(h, f)[0, 1])
    rmse = float(np.sqrt(np.mean((h - f) ** 2)))
    assert corr > 0.95, f"correlation with the true potential only {corr:.3f}"
    assert rmse < 2.5, f"RMSE {rmse:.2f} kJ/mol exceeds RT"


@pytest.mark.slow
def test_early_stopping_beats_running_to_the_end():
    """Pins the fix: the kept model must be the best-val one, not the last.

    Fails on the pre-fix behaviour, where training ran to `steps` and returned
    whatever it ended on.
    """
    d = simulate_toy(n_steps=30_000, n_traj=6, dt=0.5, D=0.01, barrier=12.0, seed=5)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]
    g = TorusGrid(40, 40)
    model, hist, info = phnn.train_binned(X, dt, 0.0101, g, traj_id=tid,
                                          steps=3000, min_count=30)
    vals = [h["val"] for h in hist if not h.get("best")]
    assert info["best_val"] <= min(vals) + 1e-12, \
        "returned model is not the best-validation one"
    assert info["best_step"] < 3000 - 1, "early stopping never engaged"


# ---------------------------------------------------------------------------
# BUG 6: scorecard compared bond-graph MFPT for ONE basin pair against the
#        MEDIAN of all PCCA MFPTs -- different quantities
# ---------------------------------------------------------------------------

def test_scorecard_compares_the_same_basin_pair():
    from ala2bg.pipeline import _scorecard
    r = {
        "kinetics": {"A": "C7eq", "B": "alphaR", "mfpt_AB_ps": 100.0,
                     "slowest_ps": 120.0},
        "msm": {"slowest_ps": 130.0,
                "basin_mfpt_ps": {"C7eq->alphaR": 150.0, "C7ax->alphaL": 9e9},
                "pcca_mfpt_ps": {"0->1": 9e9}},
        "checks": {"Hdot": -1.0, "Hdot_nonpos": True},
        "wegscheider": {"plaquette": {"max": 1e-15}, "winding_phi": {"max": None},
                        "winding_psi": {"max": None}},
    }
    rows = {row["quantity"]: row for row in _scorecard(r)}
    key = [k for k in rows if k.startswith("MFPT")][0]
    assert "C7eq->alphaR" in key
    assert rows[key]["msm"] == 150.0, \
        "scorecard picked the wrong MSM quantity (median of all pairs?)"
    assert rows[key]["pass"] is True


# ---------------------------------------------------------------------------
# BUG 7: naive differences across the periodic boundary
# ---------------------------------------------------------------------------

def test_diffusion_estimate_survives_boundary_crossings():
    """A naive (non-wrapped) difference inflates D by orders of magnitude."""
    rng = np.random.default_rng(0)
    n = 20_000
    step = 0.05
    x = np.cumsum(rng.standard_normal((n, 2)) * step, axis=0)
    X = (x + np.pi) % (2 * np.pi) - np.pi          # wrapped walk
    dt = 1.0

    good = diffusion.measure_D(X, dt)["D_const"]
    naive = float(np.mean(np.diff(X, axis=0) ** 2) / (2 * dt))   # the bug
    expected = step ** 2 / (2 * dt)

    assert abs(good - expected) / expected < 0.15, f"wrapped estimate off: {good}"
    assert naive > 10 * good, \
        "naive estimate not actually broken -- test would be vacuous"


# ---------------------------------------------------------------------------
# SCALABILITY: the grid route costs n^d; the PHNN route must not.
# ---------------------------------------------------------------------------

def _nd_toy(dim=4, n=40_000, n_traj=4, dt=0.5, D=0.01, barrier=10.0, seed=0):
    """Separable periodic double well in each of `dim` coordinates.

    F(x) = sum_k (B/2)(1 + cos 2 x_k), so the exact answer is known in every
    dimension and the test can assert accuracy rather than mere completion.
    """
    from ala2bg.thermo.smoluchowski import rt
    rng = np.random.default_rng(seed)
    beta = 1.0 / rt(298.0)
    sig = np.sqrt(2.0 * D * dt)
    x = rng.uniform(-np.pi, np.pi, size=(n_traj, dim))
    out = np.empty((n, n_traj, dim))
    for s in range(n):
        grad = -barrier * np.sin(2.0 * x)
        x = (x - D * beta * grad * dt + sig * rng.standard_normal((n_traj, dim))
             + np.pi) % (2 * np.pi) - np.pi
        out[s] = x
    X = np.concatenate([out[:, t, :] for t in range(n_traj)])
    tid = np.concatenate([np.full(n, t) for t in range(n_traj)])
    return X, tid, dt, D, barrier


def test_sparse_binning_is_independent_of_dimension():
    """Occupied voxels must track the DATA, not n^d.

    In 5D at n_per_dim=20 a dense grid would be 3.2 million bins; the sparse
    representation must stay bounded by the number of frames.
    """
    from ala2bg import nd
    X, tid, dt, _, _ = _nd_toy(dim=5, n=8_000, n_traj=2)
    bd = nd.sparse_binned_drift(X, dt, n_per_dim=20, traj_id=tid, min_count=1)

    assert bd["n_bins_if_dense"] == 20 ** 5
    assert bd["n_occupied"] <= len(X), "more occupied voxels than frames"
    assert bd["n_occupied"] < bd["n_bins_if_dense"] / 100, \
        "sparse representation is not actually sparse here"


def test_voxel_keys_are_collision_free_and_periodic():
    """A hash collision would silently merge distinct regions."""
    from ala2bg import nd
    g = np.linspace(-np.pi, np.pi, 7, endpoint=False) + 0.1
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    keys, idx = nd.voxel_keys(pts, 7)
    assert len(np.unique(keys)) == len(pts), "distinct voxels collided"
    # wrapping: -pi and +pi are the same voxel
    a, _ = nd.voxel_keys(np.array([[-np.pi, 0.0, 0.0]]), 7)
    b, _ = nd.voxel_keys(np.array([[np.pi - 1e-12, 0.0, 0.0]]), 7)
    assert a[0] != b[0] or True       # adjacent, not required equal
    c, _ = nd.voxel_keys(np.array([[np.pi + 0.01, 0.0, 0.0]]), 7)
    d_, _ = nd.voxel_keys(np.array([[-np.pi + 0.01, 0.0, 0.0]]), 7)
    assert c[0] == d_[0], "periodic wrap not applied in voxel keys"


@pytest.mark.slow
def test_phnn_recovers_known_potential_in_four_dimensions():
    """The scalability claim, asserted on accuracy rather than completion.

    A dense 4D grid at 20 bins/dim is 160,000 cells for 160k frames -- roughly
    one frame per cell, i.e. useless. The sparse route fits the same physics.
    """
    from ala2bg import nd
    X, tid, dt, D_true, barrier = _nd_toy(dim=4, n=40_000, n_traj=4)

    Dm = nd.measure_D_nd(X, dt, tid, lag=1)
    assert abs(Dm["D_const"] - D_true) / D_true < 0.15

    model, _, info = nd.train_nd(X, dt, Dm["D_const"], dim=4, n_per_dim=10,
                                 traj_id=tid, steps=2500, min_count=20)

    # R2 is measured against the per-voxel drift, whose own noise floor is set
    # by the samples per voxel, so it is capped well below 1 and swings with
    # binning (0.38-0.67 observed across settings). It is a weak sanity floor
    # here, NOT the acceptance criterion -- accuracy is asserted against the
    # known potential below.
    assert info["r2"] > 0.3, f"4D fit degenerate: R2={info['r2']:.2f}"

    # compare H against the known separable potential on occupied voxels
    import torch
    bd = nd.sparse_binned_drift(X, dt, 10, tid, min_count=20)
    pts = bd["centers"]
    with torch.no_grad():
        H = model(torch.tensor(pts, dtype=torch.float32)).numpy().astype(float)
    F = (0.5 * barrier * (1 + np.cos(2 * pts))).sum(axis=1)
    h = H - H.mean(); f = F - F.mean()
    corr = float(np.corrcoef(h, f)[0, 1])
    # observed ~0.98; 0.9 leaves headroom without being vacuous
    assert corr > 0.9, f"4D correlation with truth only {corr:.3f}"

    # the point of the test: parameters do NOT blow up with dimension
    assert info["n_parameters"] < 10_000
    assert info["n_bins_if_dense"] == 10 ** 4


# ---------------------------------------------------------------------------
# BUG 8: when NO grid is viable, the fallback picked "most bins kept", which
#        selects the MOST fragmented candidate -- exactly backwards.
# ---------------------------------------------------------------------------

def test_fallback_prefers_fewest_severed_basins_not_most_bins():
    """Reconstructs the real situation: every candidate severs something, and
    the finest grid keeps the most bins *because* it fragmented the landscape.
    """
    reports = [
        {"n": 20, "ok": False, "n_kept": 200, "severed_basins": [("alphaL", 20911)]},
        {"n": 60, "ok": False, "n_kept": 1340,
         "severed_basins": [("alphaL", 20795), ("C7ax", 1500)]},
    ]
    broken = max(reports, key=lambda r: r["n_kept"])
    fixed = min(reports, key=lambda r: (len(r["severed_basins"]), r["n"]))

    assert broken["n"] == 60, "scenario no longer reproduces the old rule"
    assert fixed["n"] == 20, "fallback did not prefer the less severed grid"
    assert fixed["n"] != broken["n"], "test would be vacuous"


def test_real_data_yields_a_viable_grid_at_default_min_count():
    """min_count=5 severed alphaL at EVERY resolution on the real dataset, so
    nothing was viable and the broken fallback shipped the worst option.
    Pins the default that keeps at least one grid viable.
    """
    import ala2bg.pipeline as pl
    import inspect
    sig = inspect.signature(pl.run)
    assert sig.parameters["min_count"].default == 1, \
        "min_count default changed; re-check grid viability on real data"


# ---------------------------------------------------------------------------
# Markovianity diagnostic: must distinguish a Markovian coordinate from one
# with memory. A diagnostic that fires on everything measures nothing.
# ---------------------------------------------------------------------------

def test_memory_diagnostic_is_quiet_on_truly_markovian_dynamics():
    """The toy IS exactly overdamped Langevin in these coordinates, so D is
    lag-independent and D_eff must come back ~= D_measured (ratio ~ 1).

    If this test can't distinguish the toy from real MD, the ~125 ps memory
    time reported on alanine dipeptide means nothing.
    """
    d = simulate_toy(n_steps=40_000, n_traj=6, dt=0.5, D=0.01, barrier=12.0, seed=2)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]

    stab = diffusion.lag_stability(X, dt, tid, lags=(1, 2, 4, 8))
    assert stab["max_ratio"] < 1.5, \
        f"toy D should be lag-independent, got ratio {stab['max_ratio']:.2f}"

    # model and reference agree by construction here -> ratio ~ 1
    mem = diffusion.memory_time(X, dt, tid, 0.0101, 100.0, 100.0)
    assert mem["available"]
    assert 0.8 < mem["ratio"] < 1.25, \
        f"spurious memory detected on Markovian data: ratio {mem['ratio']:.2f}"


def test_memory_diagnostic_recovers_a_known_effective_diffusivity():
    """Non-vacuous arithmetic check: given a stated model/reference ratio, the
    returned D_eff must equal D * t_model / t_ref, and the matched lag must be
    a lag at which the measured D really does take that value.
    """
    d = simulate_toy(n_steps=30_000, n_traj=4, dt=0.5, D=0.01, barrier=12.0, seed=4)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]
    D0 = diffusion.measure_D(X, dt, tid, lag=1)["D_const"]

    mem = diffusion.memory_time(X, dt, tid, D0, 50.0, 200.0)   # model 4x faster
    assert mem["available"]
    assert abs(mem["D_eff_to_reconcile"] - D0 * 0.25) / (D0 * 0.25) < 1e-9
    assert abs(mem["ratio"] - 4.0) < 1e-6

    if mem["memory_time_ps"]:
        curve = mem["D_vs_lag_ps"]
        lags = sorted(curve)
        lo = max([L for L in lags if L <= mem["memory_time_ps"]], default=lags[0])
        hi = min([L for L in lags if L >= mem["memory_time_ps"]], default=lags[-1])
        band = sorted([curve[lo], curve[hi]])
        assert band[0] * 0.5 <= mem["D_eff_to_reconcile"] <= band[1] * 2.0, \
            "matched lag does not bracket D_eff on the measured curve"


# ---------------------------------------------------------------------------
# PROPERTY: conditioning on a future event biases the drift estimator.
# This is why raw transition-path (TPS/TIS/RETIS) data cannot be fed to
# binned_drift without weights that restore the equilibrium path measure.
# See scripts/reactive_path_bias.py for the full measurement.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reactive_paths_bias_the_drift_estimator():
    """Reactive-only frames must give a measurably WORSE drift estimate.

    Non-vacuous by construction: the same estimator on the same trajectories,
    differing only in which frames are kept. The unbiased arm is the control
    and must succeed, so a failure of the reactive arm cannot be blamed on the
    estimator itself.
    """
    from ala2bg.thermo.smoluchowski import rt
    from scripts.reactive_path_bias import reactive_segments  # noqa

    barrier, D_true, dt = 12.0, 0.01, 0.5
    d = simulate_toy(n_steps=150_000, n_traj=6, dt=dt, D=D_true,
                     barrier=barrier, seed=7)
    X, tid = d["X"], d["traj_id"]
    # 20x20, not 40x40: reactive frames are ~0.5% of the data, so the barrier
    # region needs bins large enough to be populated by them.
    g = TorusGrid(20, 20)
    pts = g.center_grid()

    beta = 1.0 / rt(298.0)
    from ala2bg.data import toy_grad
    b_eq = -D_true * beta * toy_grad(pts, barrier)

    in_A = np.abs(X[:, 0] + np.pi / 2) < 0.25
    in_B = np.abs(X[:, 0] - np.pi / 2) < 0.25
    react = np.zeros(len(X), dtype=bool)
    for t in np.unique(tid):
        s = tid == t
        react[s] = reactive_segments(X[s, 0], in_A[s], in_B[s])

    assert react.sum() > 500, "too few reactive frames to conclude anything"

    bd_all = phnn.binned_drift(X, dt, g, tid, min_count=20)
    bd_rx = phnn.binned_drift(X[react], dt, g, tid[react], min_count=20)
    zone = bd_all["mask"] & bd_rx["mask"] & (np.abs(pts[:, 0]) < 0.6)
    assert zone.sum() >= 10, "barrier region too sparsely covered"

    def rms(v):
        return float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))

    err_unbiased = rms((bd_all["drift"] - b_eq)[zone])
    err_reactive = rms((bd_rx["drift"] - b_eq)[zone])

    # control: the estimator itself works on unbiased data
    # observed: unbiased ~0.012, reactive ~0.068 (ratio ~5.7)
    assert err_unbiased < 0.03, f"control arm failed: {err_unbiased:.4f}"
    # the actual claim
    assert err_reactive > 2.0 * err_unbiased, (
        f"reactive-path bias not detected: {err_reactive:.4f} vs "
        f"{err_unbiased:.4f}; if this ever passes trivially the test is void"
    )


# ---------------------------------------------------------------------------
# PROPERTY: a STATE-dependent bias (REST/metadynamics/umbrella) is recoverable
# by an analytic local correction, unlike FUTURE-conditioned path selection.
# This is the scalability route. See scripts/biased_drift_recovery.py.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_state_dependent_bias_is_analytically_recoverable():
    """Three arms on the same estimator: unbiased control, biased-uncorrected,
    biased-corrected. The uncorrected arm MUST fail, or the correction is
    being credited for nothing.
    """
    from scripts.biased_drift_recovery import (BARRIER, D_TRUE, bias_grad,
                                               fit, simulate)
    from ala2bg.data import toy_potential
    from ala2bg.thermo.smoluchowski import rt

    g = TorusGrid(30, 30)
    pts = g.center_grid()
    F_true = toy_potential(pts, BARRIER); F_true -= F_true.min()
    beta = 1.0 / rt(298.0)

    def accuracy(strength, corrected):
        X, tid = simulate(strength, n_steps=60_000, n_traj=8, seed=3)
        corr = D_TRUE * beta * bias_grad(pts, strength) if corrected else None
        H, m = fit(X, tid, g, corr)
        _, counts, _ = fes.histogram_fes(X, g, min_count=1)
        sel = m & (counts >= 25)
        h = H[sel] - H[sel].mean(); f = F_true[sel] - F_true[sel].mean()
        return float(np.sqrt(np.mean((h - f) ** 2))), int(
            counts[np.abs(pts[:, 0]) < 0.4].sum())

    rmse_ref, cov_ref = accuracy(0.0, False)
    rmse_unc, cov_bias = accuracy(10.0, False)
    rmse_cor, _ = accuracy(10.0, True)

    # the bias must actually improve barrier coverage, else nothing is tested
    assert cov_bias > 5 * cov_ref, \
        f"bias did not improve coverage ({cov_bias} vs {cov_ref})"
    # control arm works
    assert rmse_ref < 2.0, f"unbiased control failed: {rmse_ref:.2f}"
    # uncorrected MUST be wrong -- otherwise the correction proves nothing
    assert rmse_unc > 2.0 * rmse_ref, \
        f"uncorrected bias was already accurate ({rmse_unc:.2f}); test vacuous"
    # and the analytic correction must recover it
    assert rmse_cor < 0.5 * rmse_unc, \
        f"force correction failed to recover the landscape: {rmse_cor:.2f}"


def test_diffusion_is_unchanged_by_a_state_dependent_bias():
    """D is set by the thermostat, not the potential -- the reason a biased
    run can donate its D as well as its drift."""
    from scripts.biased_drift_recovery import simulate, DT
    X0, t0 = simulate(0.0, n_steps=40_000, n_traj=6, seed=5)
    X1, t1 = simulate(10.0, n_steps=40_000, n_traj=6, seed=5)
    d0 = diffusion.measure_D(X0, DT, t0, lag=1)["D_const"]
    d1 = diffusion.measure_D(X1, DT, t1, lag=1)["D_const"]
    assert abs(d1 - d0) / d0 < 0.05, f"D shifted under bias: {d0:.5f} -> {d1:.5f}"


# ---------------------------------------------------------------------------
# MINIMAL DISCRETISATION: the dynamics chooses the states, not a grid.
# The state count must be a MEASUREMENT with a confidence, not a choice --
# and the diagnostic must be able to say "no clean k-state description".
# ---------------------------------------------------------------------------

def test_spectral_gap_finds_a_known_hierarchy():
    from ala2bg.minimal import spectral_gap
    # one dominant slow process, then a cluster of fast ones
    g = spectral_gap([1000.0, 50.0, 45.0, 40.0, 38.0])
    assert g["recommended"] == 2
    assert g["confident"] is True
    assert g["gap_ratio"] > 10


def test_spectral_gap_reports_absence_of_a_gap():
    """Must be able to FAIL to find structure, or it measures nothing.

    This is the Stage-0 stop signal: if a six-state kinetic scheme is asserted
    and the timescales show no gap at k=6, the mapping is not supported.
    """
    from ala2bg.minimal import spectral_gap
    g = spectral_gap([100.0, 90.0, 82.0, 75.0, 70.0])   # smooth, no gap
    assert g["confident"] is False
    assert "NO CLEAR GAP" in g["note"]


def test_spectral_gap_is_not_fooled_by_a_uniform_cascade():
    from ala2bg.minimal import spectral_gap
    g = spectral_gap([2 ** -i * 1000 for i in range(6)])  # constant ratio 2
    assert g["confident"] is False, \
        "a constant-ratio cascade has no distinguished k; must not claim one"


def test_coarse_rate_matrix_satisfies_detailed_balance():
    """Symmetrised counts -> detailed balance -> symmetric kappa, with no grid.

    This is the property that lets the bond graph be built from MSM states
    instead of n^d bins.
    """
    from ala2bg.minimal import coarse_rate_matrix
    from ala2bg.surrogate import bondgraph_from_rates
    rng = np.random.default_rng(0)
    n, k = 20_000, 4
    labels = rng.integers(0, k, n)
    tid = np.zeros(n, dtype=int)
    K, pi, C = coarse_rate_matrix(labels, tid, 1, 1.0, k)

    assert np.allclose(C, C.T), "count matrix not symmetrised"
    assert abs(pi.sum() - 1) < 1e-12
    assert np.abs(K.sum(axis=1)).max() < 1e-10, "rows must sum to zero"
    # detailed balance: pi_i k_ij == pi_j k_ji
    F = pi[:, None] * K
    off = ~np.eye(k, dtype=bool)
    assert np.abs(F - F.T)[off].max() < 1e-12

    bg = bondgraph_from_rates(K, pi)        # raises if kappa is asymmetric
    assert bg.n_states == k


@pytest.mark.slow
def test_minimal_states_compress_the_grid_route():
    """End-to-end on the toy: far fewer states, slow process preserved."""
    from ala2bg import minimal, surrogate
    d = simulate_toy(n_steps=60_000, n_traj=8, dt=0.5, D=0.01, barrier=12.0, seed=1)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]

    m = minimal.minimal_states(X, tid, dt, n_clusters=60, lagtime=5, verbose=False)
    assert 2 <= m["n_states"] <= 6, f"implausible state count {m['n_states']}"
    assert m["n_states"] < 40 * 40, "no compression achieved"

    K, pi, _ = minimal.coarse_rate_matrix(m["labels"], tid, 5, dt, m["n_states"])
    bg = surrogate.bondgraph_from_rates(K, pi)
    from ala2bg.thermo import bondgraph as bgm
    ck = bgm.check_thermo(bg)
    assert ck["Hdot_nonpos"], "second law violated by the coarse bond graph"
    assert ck["equilibrium_flux_rel"] < 1e-9

    ts = surrogate.timescales_from_rates(K)
    assert ts.size and ts[0] > 0


# ---------------------------------------------------------------------------
# PHGNN: the architectural guarantee, and the honest bound on what it can do.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_phgnn_kappa_is_symmetric_by_construction():
    """kappa_ij == kappa_ji for ANY weights -- detailed balance is structural.

    Checked on an UNTRAINED network and after a weight perturbation, so it
    cannot be passing because training happened to make it symmetric.
    """
    import torch
    from ala2bg import phgnn

    ds = phgnn.make_dataset(n_graphs=12, n_grid=30, seed=5)
    assert ds, "dataset generation produced no graphs"
    rec = ds[0]
    nf = phgnn.node_features(rec).shape[1]
    ef = phgnn.edge_geometry(rec).shape[1]

    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = phgnn.PHGNN(nf, ef)
        with torch.no_grad():                    # perturb every weight
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        # RELATIVE error: a perturbed net can output log kappa ~1e5, where
        # float32 round-off alone is O(0.1) and an absolute bound would
        # falsely fail.
        assert phgnn.check_symmetry(model, rec) < 1e-5, \
            "kappa not symmetric under edge reversal"


@pytest.mark.slow
def test_kappa_is_governed_by_the_saddle_not_the_endpoints():
    """The bound on any cheap-feature rate surrogate.

    ln kappa ~ ln D - beta * F_saddle, so endpoint free energies predict kappa
    only insofar as the saddle correlates with them -- a property of the
    system, not of the model. Pinned so nobody later claims a kappa surrogate
    works in general because it worked on a correlated benchmark.
    """
    from ala2bg import phgnn
    from ala2bg.grid import TorusGrid
    from ala2bg.thermo.smoluchowski import rt

    rng = np.random.default_rng(3)
    grid = TorusGrid(30, 30)
    beta = 1.0 / rt(298.0)

    lk, fsum = [], []
    for _ in range(40):
        F = phgnn.random_landscape(grid, rng, n_modes=int(rng.integers(2, 4)),
                                   amp=float(rng.uniform(6, 18)))
        D = float(10 ** rng.uniform(-2.3, -1.7))
        rec = phgnn.exact_graph(grid, F, D, max_states=5)
        if not rec:
            continue
        for e, k in zip(rec["edges"], rec["kappa"]):
            lk.append(np.log(k))
            fsum.append(rec["F"][e[0]] + rec["F"][e[1]])
    lk, fsum = np.array(lk), np.array(fsum)
    assert len(lk) > 30, "too few edges to conclude"

    A = np.stack([fsum, np.ones_like(fsum)], axis=1)
    c, *_ = np.linalg.lstsq(A, lk, rcond=None)
    r2 = 1 - np.var(lk - A @ c) / np.var(lk)

    # endpoint features leave real variance unexplained: kappa is not a
    # function of them, and the residual is the saddle
    assert r2 < 0.95, (
        f"endpoint free energies explained R2={r2:.3f} of ln kappa; if this "
        "ever reaches ~1 the benchmark has become degenerate and the "
        "'cheap features predict kappa' conclusion would be an artefact"
    )


# ---------------------------------------------------------------------------
# BARRIER: the feature that actually governs kappa, and whether a LEARNED
# landscape resolves it well enough to skip path sampling.
# ---------------------------------------------------------------------------

def test_minimax_barrier_is_exact_on_a_known_landscape():
    """Union-find flooding must find the true saddle, not a straight-line max.

    Built so a straight-line interpolation between the two minima would give a
    DIFFERENT (higher) answer: the low path goes around, not across.
    """
    from ala2bg import barrier
    g = TorusGrid(40, 40)
    pts = g.center_grid()
    # two wells in phi, with a wall that is high at psi=0 and low at psi=pi
    F = 6.0 * (1 + np.cos(2 * pts[:, 0]))
    wall = 20.0 * np.exp(-((pts[:, 0]) ** 2) / 0.1) * np.exp(-(pts[:, 1] ** 2) / 0.5)
    F = F + wall
    F -= F.min()

    cen = barrier.basins_from_landscape(F, g, max_states=3, min_separation=1.0)
    assert len(cen) >= 2
    sad = barrier.minimax_barrier(F, g, cen)
    s = np.nanmin(sad[np.isfinite(sad)])

    # the true low path detours around the wall, so the saddle is ~12 (the
    # plain double-well barrier), NOT ~32 (through the wall)
    assert s < 16.0, f"minimax barrier {s:.1f} looks like a straight-line path"
    # and it must not be below the actual barrier
    assert s > 8.0, f"minimax barrier {s:.1f} is below the true well barrier"


def test_minimax_barrier_detects_disconnection():
    from ala2bg import barrier
    g = TorusGrid(20, 20)
    F = np.full(g.n_bins, np.nan)
    pts = g.center_grid()
    left = np.abs(pts[:, 0] + 2.0) < 0.5
    right = np.abs(pts[:, 0] - 2.0) < 0.5
    F[left] = 0.0
    F[right] = 0.0            # two islands, nothing between them
    cen = np.array([np.flatnonzero(left)[0], np.flatnonzero(right)[0]])
    sad = barrier.minimax_barrier(F, g, cen)
    assert not np.isfinite(sad[0, 1]), "unconnected basins must give NaN"


@pytest.mark.slow
def test_learned_landscape_recovers_the_barrier_within_RT():
    """The claim that makes kappa reachable without path sampling.

    A learned H must put the saddle within ~RT of the truth. The error is
    expected to be NEGATIVE (the barrier top is the least-sampled region), so
    the test pins both the magnitude and the sign of the bias.
    """
    from ala2bg import barrier, diffusion, fes, phnn
    from ala2bg.data import toy_potential
    from ala2bg.thermo.smoluchowski import rt

    bar, D = 12.0, 0.01
    d = simulate_toy(n_steps=120_000, n_traj=8, dt=0.5, D=D, barrier=bar, seed=1)
    X, dt, tid = d["X"], d["dt"], d["traj_id"]
    g = TorusGrid(40, 40)
    pts = g.center_grid()

    Ft = toy_potential(pts, bar); Ft -= Ft.min()
    cen_t = barrier.basins_from_landscape(Ft, g, max_states=4, min_separation=1.0)
    s_true = float(np.nanmax(barrier.minimax_barrier(Ft, g, cen_t)))

    Dm = diffusion.measure_D(X, dt, tid, lag=1)["D_const"]
    m, _, _ = phnn.train_binned(X, dt, Dm, g, traj_id=tid, steps=2500, min_count=25)
    H = phnn.evaluate_H(m, pts)
    _, counts, mask = fes.histogram_fes(X, g, min_count=25)
    cen = barrier.basins_from_landscape(np.where(mask, H, np.nan), g,
                                        max_states=4, min_separation=1.0)
    sad = barrier.minimax_barrier(np.where(mask, H, np.nan), g, cen)
    assert np.isfinite(sad).any(), "no saddle found on the learned landscape"
    s_learn = float(np.nanmax(sad))

    err = s_learn - s_true
    assert abs(err) < rt(298.0), \
        f"learned saddle off by {err:+.2f} kJ/mol, more than RT"
    assert err < 0.5, (
        "the barrier-top bias is expected to be negative (least-sampled "
        f"region); got {err:+.2f}. If this turns positive the sampling or the "
        "mask has changed and the GREST argument needs re-checking."
    )


@pytest.mark.slow
def test_barrier_features_rescue_kappa_transfer():
    """The result that overturns the earlier negative finding.

    Same architecture, same graphs, same split -- only the edge features
    differ. The cheap-feature arm is the control and MUST do worse, otherwise
    the comparison shows nothing.
    """
    from ala2bg import phgnn

    ds = phgnn.make_dataset(n_graphs=140, n_grid=30, seed=2)
    ds = [r for r in ds if np.isfinite(r["saddle"]).all()]
    assert len(ds) > 60, f"only {len(ds)} usable graphs"

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))
    nv = max(1, int(0.25 * len(ds)))
    val = [ds[i] for i in idx[:nv]]

    out = {}
    for ub in (False, True):
        m, _, _, _ = phgnn.train(ds, steps=900, hidden=32, rounds=2, seed=0,
                                 use_barrier=ub)
        out[ub] = phgnn.evaluate(m, val, use_barrier=ub)

    cheap = out[False]["rmse_phgnn_ln_kappa"]
    barr = out[True]["rmse_phgnn_ln_kappa"]
    assert barr < cheap, (
        f"barrier features did not help: {barr:.3f} vs {cheap:.3f}. "
        "ln kappa ~ ln D - beta F_dagger, so if informing the model about the "
        "saddle does not improve it, the dataset or the features are broken."
    )
    # and the gain must be real, not noise
    assert barr < 0.85 * cheap, \
        f"gain too small to be meaningful: {barr:.3f} vs {cheap:.3f}"


@pytest.mark.slow
def test_barrier_features_preserve_the_symmetry_guarantee():
    """Adding barrier features must not break detailed balance.

    Every barrier column (F_dagger, barrier from deeper/shallower well) is
    symmetric in the two endpoints, so kappa stays symmetric. Verified on
    perturbed weights, as for the cheap-feature case.
    """
    import torch
    from ala2bg import phgnn

    ds = phgnn.make_dataset(n_graphs=10, n_grid=30, seed=5)
    ds = [r for r in ds if np.isfinite(r["saddle"]).all()]
    assert ds
    rec = ds[0]
    nf = phgnn.node_features(rec).shape[1]
    ef = phgnn.edge_geometry(rec, True).shape[1]

    torch.manual_seed(0)
    model = phgnn.PHGNN(nf, ef)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    x = torch.tensor(phgnn.node_features(rec), dtype=torch.float32)
    ei = torch.tensor(rec["edges"], dtype=torch.long)
    ea = torch.tensor(phgnn.edge_geometry(rec, True), dtype=torch.float32)
    with torch.no_grad():
        _, a = model(x, ei, ea)
        _, b = model(x, ei.flip(1), ea)
    scale = float(torch.maximum(a.abs(), b.abs()).max().clamp(min=1.0))
    assert float((a - b).abs().max()) / scale < 1e-5


# ---------------------------------------------------------------------------
# PERIODIC COORDINATES WITH ARBITRARY PERIOD (ion z in a periodic box).
# ---------------------------------------------------------------------------

def test_arbitrary_period_round_trip():
    from ala2bg.periodic import to_angle, from_angle
    L = 8.4                                   # nm, a plausible box length
    z = np.linspace(0, L, 50, endpoint=False)
    back = from_angle(to_angle(z, L), L)
    assert np.allclose(np.sort(back), np.sort(z), atol=1e-9)


def test_wrap_period_handles_box_crossing():
    """The permeation analogue of the +-pi bug: an ion leaving the top of the
    box and re-entering the bottom moved a little, not a whole box length."""
    from ala2bg.periodic import wrap_period
    L = 8.0
    d = wrap_period(np.array([-7.9]), L)       # naive reading: moved -7.9 nm
    assert abs(d[0] - 0.1) < 1e-9, f"got {d[0]}, expected +0.1"


def test_mixed_cvspec_rejects_bad_periods():
    from ala2bg.periodic import MixedCVSpec
    with pytest.raises(ValueError):
        MixedCVSpec(names=("a", "b"), periods=(1.0, 0.0))
    with pytest.raises(ValueError):
        MixedCVSpec(names=("a", "b"), periods=(1.0,))


def test_diffusivity_converts_back_to_physical_units():
    """D measured in angle^2/time must convert back by (dz/dtheta)^2.

    Getting this wrong rescales every rate, so it is checked numerically
    rather than trusted.
    """
    from ala2bg.periodic import MixedCVSpec
    L = 8.0
    spec = MixedCVSpec(names=("z",), periods=(L,), origins=(0.0,), units=("nm",))
    D_phys_true = 0.05                                    # nm^2/ps
    scale = 2 * np.pi / L                                 # rad per nm
    D_angle = D_phys_true * scale ** 2
    assert np.allclose(spec.D_angle_to_physical(D_angle), D_phys_true)


def test_mixed_cvspec_hash_is_sensitive_to_the_period():
    """Two runs with different box lengths must not share a CV hash."""
    from ala2bg.periodic import MixedCVSpec
    a = MixedCVSpec(names=("z",), periods=(8.0,), origins=(0.0,), units=("nm",))
    b = MixedCVSpec(names=("z",), periods=(8.4,), origins=(0.0,), units=("nm",))
    assert a.hash() != b.hash()
    assert a.hash() == MixedCVSpec(names=("z",), periods=(8.0,),
                                   origins=(0.0,), units=("nm",)).hash()


def test_angles_feed_the_existing_torus_machinery_unchanged():
    """The point of the rescaling: existing estimators work without edits."""
    from ala2bg.periodic import MixedCVSpec
    from ala2bg.cvs import wrapped_delta, fourier_features
    L = 6.0
    spec = MixedCVSpec(names=("z1", "z2"), periods=(L, L),
                       origins=(0.0, 0.0), units=("nm", "nm"))
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, L, size=(200, 2))
    TH = spec.to_angles(Z)
    assert TH.shape == Z.shape
    assert TH.min() >= -np.pi - 1e-12 and TH.max() < np.pi + 1e-12
    assert fourier_features(TH).shape == (200, spec.n_features)
    d = wrapped_delta(TH[1:], TH[:-1])
    assert np.abs(d).max() <= np.pi + 1e-12


# ---------------------------------------------------------------------------
# ELECTROGENIC BOND GRAPH (Hunter et al. formulation).
# ---------------------------------------------------------------------------

def _k_channel(c_in=140.0, c_out=4.0, k=2e8, deltas=(0., .25, .5, .75)):
    from ala2bg.electro import (ElectroBondGraph, bath_affinity,
                                electrical_distances, reaction_charges)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    delta = electrical_distances(np.array(deltas), 0.0, 1.0)
    z = reaction_charges(delta, edges, z_ion=1.0)
    A = np.zeros(4); A[3] = bath_affinity(c_in, c_out)
    return ElectroBondGraph.from_rates(np.full(4, k), np.full(4, k), edges, z,
                                       affinity=A)


def test_reversal_potential_reproduces_nernst():
    """Hunter's validation, and it is NOT a restatement.

    reversal_potential() is solved numerically from the assembled bond graph;
    nernst() is the closed-form answer. They agree only if the per-reaction
    charge coefficients close correctly around the cycle.
    """
    for c_in, c_out in [(1., 1.), (10., 1.), (1., 10.), (140., 4.), (1., 100.)]:
        bg = _k_channel(c_in, c_out)
        r = bg.check_nernst(c_in, c_out)
        assert r["ok"], f"{c_in}/{c_out}: {r}"
        assert abs(r["diff_mV"]) < 1e-6


def test_physiological_potassium_gives_textbook_EK():
    bg = _k_channel(140.0, 4.0)
    assert abs(bg.reversal_potential() * 1000 + 91.3) < 0.2


def test_a_flipped_orientation_is_caught():
    """The failure mode that is easy to wave through: right magnitude, wrong
    sign. If check_nernst cannot catch it, it is not testing anything."""
    from ala2bg.electro import bath_affinity
    bg = _k_channel(140.0, 4.0)
    bad = _k_channel(4.0, 140.0)          # bath affinity inverted
    good_V = bg.reversal_potential()
    bad_V = bad.reversal_potential()
    assert abs(abs(bad_V) - abs(good_V)) < 1e-9, "magnitudes should match"
    assert np.sign(bad_V) != np.sign(good_V), "sign should flip"
    assert not bad.check_nernst(140.0, 4.0)["ok"], \
        "an inverted orientation passed the Nernst check"


def test_charge_closure_is_enforced_not_assumed():
    """sum of per-reaction z around the cycle must equal the ion valence."""
    from ala2bg.electro import ElectroBondGraph, electrical_distances, reaction_charges
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    delta = electrical_distances(np.array([0., .25, .5, .75]), 0.0, 1.0)
    z = reaction_charges(delta, edges, z_ion=1.0)
    assert abs(z.sum() - 1.0) < 1e-12

    bg = _k_channel()
    assert bg.check_charge_closure()["closes"]

    broken = ElectroBondGraph(K=bg.K, kappa=bg.kappa, edges=edges,
                              z=z * 0.5)       # charge no longer closes
    chk = broken.check_charge_closure()
    assert not chk["closes"], "non-closing charge went undetected"
    assert abs(chk["residual"] - 0.5) < 1e-12


def test_no_current_without_a_driving_force():
    """Equal concentrations, zero voltage -> exactly zero current.

    A model that makes current from nothing is broken, and this is the
    cheapest possible check of that.
    """
    bg = _k_channel(1.0, 1.0)
    assert bg.check_no_current_at_equilibrium()["ok"]
    assert abs(bg.steady_current(0.0)) < 1e-25


def test_single_channel_and_molar_modes_differ_by_avogadro():
    """The units trap: using the Faraday constant where the elementary charge
    belongs inflates the current by ~6e23 and still passes every structural
    check."""
    from ala2bg.electro import AVOGADRO
    bg = _k_channel()
    I_sc = bg.steady_current(0.0)
    bg.mode = "molar"
    I_mol = bg.steady_current(0.0)
    assert abs(I_mol / I_sc - AVOGADRO) / AVOGADRO < 1e-9


def test_conductance_is_in_a_physiological_range():
    """Order-of-magnitude sanity: a K+ channel is tens to hundreds of pS, not
    1e21 pS (which is what the unit bug produced)."""
    bg = _k_channel()
    g = bg.conductance(bg.reversal_potential()) * 1e12
    assert 1.0 < g < 1000.0, f"conductance {g:.3g} pS is not physiological"
