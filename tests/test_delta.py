"""The delta mapping: state charges for states that have no position.

Each test reconstructs the tempting wrong mapping -- delta of the mean ion
position per state -- and asserts that it differs from the right answer, so
the test cannot pass on the broken construction.
"""

import numpy as np
import pytest

from ala2bg.delta import (delta_profile_fn, frame_charges, state_charges,
                          reaction_charges_from_states)
from ala2bg.electro import ElectroBondGraph, Species, Transit


def _linear_delta(lo=-10.0, hi=10.0):
    z = np.linspace(lo, hi, 201)
    return delta_profile_fn(z, (z - lo) / (hi - lo))


def _filter_delta():
    """Nonlinear: a narrow filter in [-5, 5] carries 90% of the drop."""
    from ala2bg import kcsa
    z = np.linspace(-15.0, 15.0, 1201)
    r = np.where(np.abs(z) < 5.0, 1.2, 5.0)
    return delta_profile_fn(z, kcsa.electrical_distance_from_geometry(z, z, r))


def test_two_ion_state_carries_the_sum_not_the_mean_position():
    """Ions at delta 0.25 and 0.75 hold one full charge; delta(mean z) = 0.5."""
    f = _linear_delta()
    n = 400
    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.normal(-5.0, 0.2, n), rng.normal(5.0, 0.2, n)])
    out = state_charges(Z, np.zeros(n, int), 1, f)
    assert out["Q"][0] == pytest.approx(1.0, abs=0.01)

    naive = float(f(np.nanmean(Z)))
    assert abs(naive - out["Q"][0]) > 0.4


def test_knock_on_moves_twice_the_charge_of_a_single_hop():
    """Two ions advancing one site each carry 2x the single-ion charge."""
    f = _linear_delta()
    sites = np.array([-7.5, -2.5, 2.5, 7.5])        # delta .125 .375 .625 .875
    n = 200
    A = np.tile(sites[[0, 2]], (n, 1))              # ions at S_a, S_c
    B = np.tile(sites[[1, 3]], (n, 1))              # both advanced one site
    Z = np.vstack([A, B])
    lab = np.r_[np.zeros(n, int), np.ones(n, int)]
    Q = state_charges(Z, lab, 2, f)["Q"]
    z = reaction_charges_from_states(Q, [[0, 1]])
    single_hop = float(f(sites[1]) - f(sites[0]))
    assert z[0] == pytest.approx(2 * single_hop, abs=1e-9)

    naive = float(f(B.mean()) - f(A.mean()))
    assert naive == pytest.approx(single_hop, abs=1e-9)   # the wrong answer


def test_nonlinear_profile_averages_delta_not_position():
    """Jensen: with a filter-concentrated drop, <delta(z)> != delta(<z>)."""
    f = _filter_delta()
    rng = np.random.default_rng(1)
    # a state straddling the filter's outer mouth: half in the filter, half in
    # the wide vestibule where delta barely changes
    Z = np.r_[rng.normal(3.0, 0.3, 2000), rng.normal(10.0, 0.3, 2000)][:, None]
    Q = state_charges(Z, np.zeros(len(Z), int), 1, f)["Q"][0]
    assert Q == pytest.approx(float(np.mean(f(Z))), abs=1e-12)
    assert abs(Q - float(f(Z.mean()))) > 0.05


def test_state_charge_is_permutation_invariant():
    """Relabelling ions is physically nothing; Q must not move at all."""
    f = _filter_delta()
    rng = np.random.default_rng(2)
    Z = rng.uniform(-12, 12, size=(500, 3))
    Z[rng.random(Z.shape) < 0.1] = np.nan              # absent ions
    P = np.array([rng.permutation(row) for row in Z])
    q0, n0 = frame_charges(Z, f)
    q1, n1 = frame_charges(P, f)
    np.testing.assert_array_equal(n0, n1)
    np.testing.assert_allclose(q0, q1, atol=1e-12)
    # and a per-ion feature is NOT invariant, so the test can tell
    assert np.max(np.abs(f(np.nan_to_num(Z[:, 0], nan=-20.0))
                         - f(np.nan_to_num(P[:, 0], nan=-20.0)))) > 0.1


def test_we_weights_enter_as_weights():
    """Weighted-ensemble walker weights must weight the state average."""
    f = _linear_delta()
    Z = np.array([[-5.0], [5.0]])                     # delta .25 and .75
    w = np.array([0.9, 0.1])
    Q = state_charges(Z, [0, 0], 1, f, weights=w)["Q"][0]
    assert Q == pytest.approx(0.9 * 0.25 + 0.1 * 0.75)
    assert abs(Q - state_charges(Z, [0, 0], 1, f)["Q"][0]) > 0.1


def test_closure_follows_from_stoichiometry_for_any_state_charges():
    """Around the E -> S_in ... S_out -> E cycle, sum z = 1 whatever Q is.

    Dropping the bath-species term leaves a state function, which telescopes
    to ZERO around the cycle -- the bug-12 failure mode.
    """
    rng = np.random.default_rng(3)
    n = 6                                              # 5 sites + E
    edges = np.array([[i, i + 1] for i in range(n - 2)]
                     + [[n - 2, n - 1], [n - 1, 0]])
    tr = [Transit(n - 2, "K_ref", "K_out"), Transit(n - 1, "K_in", "K_ref")]
    sd = {"K_in": 0.0, "K_out": 1.0, "K_ref": None}
    for _ in range(20):
        Q = rng.uniform(0, 2, n)
        z = reaction_charges_from_states(Q, edges, tr, sd)
        assert z.sum() == pytest.approx(1.0, abs=1e-12)
        assert reaction_charges_from_states(Q, edges).sum() == \
            pytest.approx(0.0, abs=1e-12)


def test_bath_steps_can_carry_charge_and_nernst_still_holds():
    """Unrenormalised geometric delta: binding from the cavity traverses part
    of the field. The assembled model must still reverse at Nernst."""
    f = _filter_delta()
    site_z = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    Q = np.r_[f(site_z), 0.0]                          # E holds no ion
    n = len(Q)
    edges = np.array([[i, i + 1] for i in range(n - 2)]
                     + [[n - 2, n - 1], [n - 1, 0]])
    tr = [Transit(n - 2, "K_ref", "K_out"), Transit(n - 1, "K_in", "K_ref")]
    z = reaction_charges_from_states(
        Q, edges, tr, {"K_in": 0.0, "K_out": 1.0, "K_ref": None})
    assert z[-1] > 0.05 and z[-2] > 0.05               # bath steps do carry
    assert z.sum() == pytest.approx(1.0, abs=1e-12)

    k = np.full(len(edges), 1e8)
    sp = {"K_in": Species("K_in", 1.0, 140.0),
          "K_out": Species("K_out", 1.0, 4.0),
          "K_ref": Species("K_ref", 1.0, 100.0)}
    bg = ElectroBondGraph.from_rates(k, k, edges, z, species=sp, transits=tr)
    chk = bg.check_nernst(140.0, 4.0)
    assert chk["ok"], chk


def test_unnamed_bath_species_is_an_error_not_a_zero():
    edges = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="no electrical position"):
        reaction_charges_from_states([0.0, 0.5], edges,
                                     [Transit(1, "K_in", "K_out")],
                                     {"K_in": 0.0})


def test_unvisited_state_is_refused():
    f = _linear_delta()
    out = state_charges(np.zeros((10, 1)), np.zeros(10, int), 2, f)
    assert out["unvisited_states"] == [1]
    with pytest.raises(ValueError, match="never visited"):
        reaction_charges_from_states(out["Q"], [[0, 1]])


def test_count_purity_flags_a_state_mixing_occupancies():
    """A single-ion CV lumps 1- and 2-ion configurations into one state."""
    f = _linear_delta()
    Z = np.array([[0.0, np.nan]] * 50 + [[0.0, 5.0]] * 50)
    out = state_charges(Z, np.zeros(100, int), 1, f)
    assert out["count_purity"][0] == pytest.approx(0.5)
    pure = state_charges(Z[:50], np.zeros(50, int), 1, f)
    assert pure["count_purity"][0] == pytest.approx(1.0)


def test_profile_must_be_normalised():
    z = np.linspace(0, 1, 11)
    with pytest.raises(ValueError, match="0 -> 1"):
        delta_profile_fn(z, 0.5 * z)
