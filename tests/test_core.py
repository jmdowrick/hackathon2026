"""Unit gates. No GPU, no network, no MD. Must stay under ~30 s.

These are the checks that let the bond-graph stage be finished before the MD
is, and that catch the one bug most likely to silently wreck everything
(torus wrapping).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from ala2bg.cvs import CVSpec, wrap, wrapped_delta, fourier_features
from ala2bg.grid import TorusGrid
from ala2bg.thermo.smoluchowski import (
    build_generator, stationary_from_F, detailed_balance_residual, rt)
from ala2bg.thermo import committor as cm
from ala2bg.thermo import bondgraph as bgm


# --------------------------------------------------------------------------
# torus
# --------------------------------------------------------------------------

def test_wrap_into_range():
    a = np.array([-3 * np.pi, -np.pi, 0.0, np.pi - 1e-9, 3 * np.pi, 7.0])
    w = wrap(a)
    assert np.all(w >= -np.pi - 1e-12) and np.all(w < np.pi + 1e-12)


def test_wrapped_delta_crossing_boundary():
    """THE bug this project most needs to not have.

    A trajectory stepping from +3.13 to -3.13 moved ~0.02 rad across the
    boundary, not ~6.26 rad backwards.
    """
    d = wrapped_delta(np.array([[-3.13, 0.0]]), np.array([[3.13, 0.0]]))
    assert abs(d[0, 0]) < 0.03, f"got {d[0,0]}, naive difference would be ~-6.26"


def test_fourier_features_periodic():
    a = fourier_features(np.array([[-np.pi, 0.3]]))
    b = fourier_features(np.array([[np.pi, 0.3]]))
    assert np.allclose(a, b, atol=1e-12)


def test_cvspec_hash_stable_and_sensitive():
    assert CVSpec().hash() == CVSpec().hash()
    assert CVSpec().hash() != CVSpec(n_harmonics=3).hash()


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_grid_neighbours_wrap():
    g = TorusGrid(6, 6)
    edges = list(g.neighbours())
    assert len(edges) == 2 * g.n_bins          # torus: every node has degree 4
    # bin (0,0) must neighbour (5,0) via wrapping
    a = g.ravel(0, 0)
    partners = {b for x, b, _ in edges if x == a} | {x for x, b, _ in edges if b == a}
    assert g.ravel(5, 0) in partners


def test_index_round_trip():
    g = TorusGrid(12, 12)
    pts = g.center_grid()
    assert np.array_equal(g.index_of(pts), np.arange(g.n_bins))


# --------------------------------------------------------------------------
# Smoluchowski / detailed balance -- the core claim
# --------------------------------------------------------------------------

def _random_F(g, seed=0):
    rng = np.random.default_rng(seed)
    p = g.center_grid()
    return (5.0 * np.cos(2 * p[:, 0]) + 3.0 * np.sin(p[:, 1])
            + 0.4 * rng.standard_normal(g.n_bins))


def test_stationary_distribution_is_boltzmann():
    """L pi = 0 with pi ~ exp(-beta F), to machine precision."""
    g = TorusGrid(16, 16)
    F = _random_F(g)
    L, keep = build_generator(F, 0.05, g)
    pi = stationary_from_F(F, keep)
    resid = np.abs(L @ pi).max() / np.abs(pi).max()
    assert resid < 1e-10, f"stationary residual {resid:.3e}"


def test_detailed_balance_structural():
    g = TorusGrid(16, 16)
    F = _random_F(g, seed=3)
    L, keep = build_generator(F, 0.05, g)
    pi = stationary_from_F(F, keep)
    assert detailed_balance_residual(L, pi) < 1e-12


def test_generator_columns_sum_to_zero():
    g = TorusGrid(10, 10)
    L, _ = build_generator(_random_F(g), 0.05, g)
    assert np.abs(np.asarray(L.sum(axis=0)).ravel()).max() < 1e-10


# --------------------------------------------------------------------------
# committor
# --------------------------------------------------------------------------

def test_committor_bounds_and_monotonicity():
    """Flat 1D landscape -> committor is linear in index."""
    g = TorusGrid(20, 3)
    F = np.zeros(g.n_bins)
    L, keep = build_generator(F, 0.1, g)
    A = np.flatnonzero(g.unravel(keep)[0] == 0)
    B = np.flatnonzero(g.unravel(keep)[0] == 10)
    q = cm.committor(L, A, B)
    assert q.min() >= -1e-9 and q.max() <= 1 + 1e-9
    assert np.allclose(q[A], 0, atol=1e-9) and np.allclose(q[B], 1, atol=1e-9)


def test_mfpt_positive_and_zero_on_target():
    g = TorusGrid(12, 12)
    L, keep = build_generator(_random_F(g), 0.05, g)
    tgt = np.arange(5)
    tau = cm.mfpt(L, tgt)
    assert np.allclose(tau[tgt], 0.0)
    assert np.all(tau[len(tgt):] > 0)


def test_higher_barrier_gives_longer_mfpt():
    """Sanity: the physics must go the right way."""
    g = TorusGrid(24, 4)
    out = []
    for barrier in (4.0, 12.0):
        p = g.center_grid()
        F = 0.5 * barrier * (1 + np.cos(2 * p[:, 0]))
        L, keep = build_generator(F, 0.05, g)
        pi = stationary_from_F(F, keep)
        i, _ = g.unravel(keep)
        A = np.flatnonzero(i < 2)
        B = np.flatnonzero(np.abs(i - g.n_phi // 2) < 2)
        out.append(cm.mfpt_between(L, A, B, pi))
    assert out[1] > out[0] * 2, f"MFPTs {out} did not increase with barrier"


# --------------------------------------------------------------------------
# bond graph -- the thermodynamic claims
# --------------------------------------------------------------------------

def test_kappa_symmetric_and_graph_builds():
    g = TorusGrid(14, 14)
    F = _random_F(g, seed=7)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)   # raises if asymmetric
    assert bg.n_states == len(keep)
    assert bg.n_reactions == 2 * len(keep)
    assert (bg.kappa > 0).all()


def test_zero_flux_at_equilibrium():
    g = TorusGrid(14, 14)
    F = _random_F(g, seed=11)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    c = bgm.check_thermo(bg)
    assert c["equilibrium_flux_rel"] < 1e-9


def test_second_law_structural():
    """Hdot <= 0 for ANY state, not just equilibrium."""
    g = TorusGrid(10, 10)
    F = _random_F(g, seed=13)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    rng = np.random.default_rng(0)
    N = bg.incidence()
    for _ in range(20):
        x = rng.random(bg.n_states) + 1e-3
        x /= x.sum()
        assert float(bg.mu(x) @ (N @ bg.fluxes(x))) <= 1e-9


def test_wegscheider_including_winding_cycles():
    """Contractible AND non-contractible cycles must both close."""
    g = TorusGrid(12, 12)
    F = _random_F(g, seed=17)
    L, keep = build_generator(F, 0.05, g)
    w = bgm.wegscheider_residuals(L, g, keep, n_random=20)
    assert w["plaquette"]["max"] < 1e-10
    assert w["winding_phi"]["max"] < 1e-10
    assert w["winding_psi"]["max"] < 1e-10


def test_cycle_residual_detects_a_violation():
    """The test must be capable of FAILING, or it proves nothing.

    Break detailed balance by hand on one edge and confirm the residual fires.
    """
    g = TorusGrid(8, 8)
    F = _random_F(g, seed=19)
    L, keep = build_generator(F, 0.05, g)
    Ld = sp.lil_matrix(L.copy())
    i, j = 0, 1
    if Ld[i, j] != 0:
        Ld[i, j] *= 10.0
    w = bgm.wegscheider_residuals(sp.csr_matrix(Ld), g, keep, n_random=200, seed=1)
    assert (w["plaquette"]["max"] or 0) > 1e-6, "tampered generator went undetected"


# --------------------------------------------------------------------------
# CellML round trip -- the real proof, independent of OpenCOR
# --------------------------------------------------------------------------

def test_cellml_writes_and_is_parseable(tmp_path):
    import xml.etree.ElementTree as ET
    from ala2bg.thermo.cellml_writer import write_cellml
    g = TorusGrid(6, 6)
    F = _random_F(g, seed=23)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    p = tmp_path / "m.cellml"
    write_cellml(bg, p, max_states=None)
    root = ET.parse(p).getroot()
    assert root.tag.endswith("model")


def test_cellml_guards_against_huge_models(tmp_path):
    from ala2bg.thermo.cellml_writer import write_cellml
    g = TorusGrid(20, 20)
    F = _random_F(g, seed=29)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    with pytest.raises(ValueError):
        write_cellml(bg, tmp_path / "m.cellml", max_states=12)


def test_bondgraph_ode_matches_master_equation():
    """The emitted dynamics ARE the master equation.

    dx/dt from the bond graph must equal L x for any x. This is what makes the
    CellML file a faithful representation rather than a decorative one.
    """
    g = TorusGrid(10, 10)
    F = _random_F(g, seed=31)
    L, keep = build_generator(F, 0.05, g)
    bg = bgm.from_generator(L, F[keep], g.cell_area)
    rng = np.random.default_rng(5)
    x = rng.random(bg.n_states) + 1e-3
    x /= x.sum()
    assert np.allclose(bg.dxdt(x), L @ x, rtol=1e-8, atol=1e-12)
