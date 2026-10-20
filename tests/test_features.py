"""Permutation-invariant features and the gramicidin A frame.

Each test reconstructs the broken alternative (raw per-ion coordinates, no
radial gate, sorted-with-padding, no minimum image, HETATM dropped, off-axis
origin) and asserts it differs, so the test cannot pass on it.
"""

import os

import numpy as np
import pytest

from ala2bg.features import (Channel, in_channel, to_pore_frame,
                             basis_centres, site_occupancy, invariant_features)

CH = Channel(-12.0, 12.0, r_max=4.0, width=0.5)


def _ions(n_frames=300, n_ions=4, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-20, 20, (n_frames, n_ions))
    r = rng.uniform(0, 8, (n_frames, n_ions))
    absent = rng.random(z.shape) < 0.1
    z[absent] = np.nan
    r[absent] = np.nan
    return z, r


def test_features_are_permutation_invariant_and_raw_coordinates_are_not():
    z, r = _ions()
    rng = np.random.default_rng(1)
    perm = np.array([rng.permutation(z.shape[1]) for _ in range(len(z))])
    zp = np.take_along_axis(z, perm, 1)
    rp = np.take_along_axis(r, perm, 1)
    X0, _ = invariant_features(z, r, CH)
    X1, _ = invariant_features(zp, rp, CH)
    np.testing.assert_allclose(X0, X1, atol=1e-12)
    raw = np.nan_to_num(z) - np.nan_to_num(zp)
    assert np.max(np.abs(raw)) > 10.0


def test_radial_gate_excludes_bulk_ion_at_channel_z():
    """Bug 2: an ion at z = 0 but 15 A off-axis is not in the pore."""
    z = np.array([[0.0, 0.0]])
    r = np.array([[0.5, 15.0]])
    X, names = invariant_features(z, r, CH)
    n = X[0, names.index("n_channel")]
    assert n == pytest.approx(1.0, abs=1e-3)
    ungated = invariant_features(z, np.zeros_like(r), CH)[0]
    assert ungated[0, names.index("n_channel")] == pytest.approx(2.0, abs=1e-3)


def test_channel_without_radial_gate_is_refused():
    with pytest.raises(ValueError, match="radial gate"):
        Channel(-10, 10, r_max=0.0)


def test_delta_charges_respect_the_same_radial_gate():
    from ala2bg.delta import delta_profile_fn, frame_charges
    zp = np.linspace(-12, 12, 101)
    f = delta_profile_fn(zp, (zp + 12) / 24)
    z = np.array([[0.0, 6.0]])
    r = np.array([[0.5, 15.0]])                        # second is bulk
    q_gated, n_gated = frame_charges(z, f, r_ions=r, r_max=4.0)
    q_raw, n_raw = frame_charges(z, f)
    assert n_gated[0] == 1 and q_gated[0] == pytest.approx(0.5)
    assert n_raw[0] == 2 and q_raw[0] == pytest.approx(1.25)
    with pytest.raises(ValueError, match="both"):
        frame_charges(z, f, r_ions=r)


def test_occupancies_partition_the_ion_count():
    z, r = _ions(seed=2)
    c = basis_centres(CH, 1.0)
    occ = site_occupancy(z, r, CH, c)
    n = in_channel(z, r, CH).sum(axis=1)
    np.testing.assert_allclose(occ.sum(axis=1), n, atol=1e-12)
    deep = site_occupancy(np.array([[0.3]]), np.array([[0.2]]), CH, c)
    # soft edge: 3.8 A inside a 0.5 A-wide radial edge is sigmoid(7.6)
    assert deep.sum() == pytest.approx(1.0, abs=1e-3)


def test_features_are_continuous_through_channel_entry():
    """One ion walks in from the bath in 0.05 A steps while another sits in
    the pore. Sorted-positions-with-padding teleports at entry; occupancies
    must not."""
    steps = np.arange(-18.0, -6.0, 0.05)
    z = np.column_stack([steps, np.full_like(steps, 5.0)])
    r = np.full_like(z, 0.5)
    X, _ = invariant_features(z, r, CH)
    assert np.max(np.abs(np.diff(X, axis=0))) < 0.15

    pad = -15.0
    m = in_channel(z, r, CH, soft=False).astype(bool)
    sorted_pad = np.sort(np.where(m, z, pad), axis=1)
    assert np.max(np.abs(np.diff(sorted_pad, axis=0))) > 2.0


def test_minimum_image_keeps_an_ion_in_the_pore():
    """Bug 7's class: an ion wrapped to the far side of the box is still at
    the pore, not a box length away."""
    box = np.array([60.0, 60.0, 80.0])
    origin = np.array([30.0, 30.0, 2.0])
    xyz = np.array([[[30.5, 30.0, 2.0 - 1.0 + 80.0]]])   # z wrapped by +Lz
    z, r = to_pore_frame(xyz, origin, [0, 0, 1], box=box)
    assert z[0, 0] == pytest.approx(-1.0) and r[0, 0] == pytest.approx(0.5)
    z_naive, _ = to_pore_frame(xyz, origin, [0, 0, 1])
    assert z_naive[0, 0] == pytest.approx(79.0)


def test_q_feature_matches_delta_module():
    from ala2bg.delta import delta_profile_fn, frame_charges
    zp = np.linspace(-12, 12, 101)
    f = delta_profile_fn(zp, (zp + 12) / 24)
    hard = Channel(-12, 12, 4.0, width=0.0)
    z, r = _ions(seed=3)
    X, names = invariant_features(z, r, hard, delta_fn=f)
    q, _ = frame_charges(z, f, r_ions=r, r_max=4.0)
    np.testing.assert_allclose(X[:, names.index("Q")], q, atol=1e-12)


# --- gramicidin A ---------------------------------------------------------

needs_1jno = pytest.mark.skipif(not os.path.exists("data/1JNO.pdb"),
                                reason="1JNO not cached")


@needs_1jno
def test_ga_frame_is_the_dimer_axis():
    from ala2bg import gramicidin as ga
    g = ga.load_gramicidin()
    assert g["axis_anisotropy"] > 2.0
    assert sorted(g["nterm_z_A"]) == pytest.approx([-0.96, 0.96], abs=0.1)
    assert g["cterm_z_A"][0] > 10.0 and g["cterm_z_A"][1] < -10.0
    r = g["radius_A"]
    # the dimer's two-fold symmetry is an independent check of the frame
    assert np.nanmax(np.abs(r - r[::-1])) < 0.05


@needs_1jno
def test_ga_lumen_is_not_clipped_by_an_off_axis_origin():
    """An origin at the formyl-valine midpoint, off the helix axis, made
    interior slabs read at the 0.05 A clip floor."""
    from ala2bg import gramicidin as ga, pdb
    g = ga.load_gramicidin()
    z, r = g["radius_z_A"], g["radius_A"]
    interior = np.abs(z) < 10.0
    assert np.nanmin(r[interior]) > 0.5

    s, _ = pdb.read_pdb(g["path"], keep_chains=["A", "B"])
    junction = np.mean([s.xyz[s.select(chain=c, resname="FVA")].mean(axis=0)
                        for c in "AB"], axis=0)
    edges = np.linspace(-20, 20, len(z) + 1)
    bad = pdb.radius_profile(s, junction, g["axis"], edges,
                             wall_mask=pdb.non_solvent(s))
    assert np.nanmin(bad[interior]) < 0.1


@needs_1jno
def test_ga_wall_includes_d_residues():
    """Half of GA is HETATM. Dropping it widens the measured pore."""
    from ala2bg import gramicidin as ga, pdb
    g = ga.load_gramicidin()
    assert g["n_hetatm_in_wall"] > 100
    s, _ = pdb.read_pdb(g["path"], keep_chains=["A", "B"])
    edges = np.linspace(-20, 20, len(g["radius_A"]) + 1)
    atom_only = pdb.radius_profile(s, g["origin"], g["axis"], edges)
    assert np.nanmax(atom_only - g["radius_A"]) > 0.5
