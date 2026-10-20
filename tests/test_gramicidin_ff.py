"""Gramicidin A force field: the D-residues, the caps, and nothing silent.

Each test is shown able to fail: the mirror test against a chiral torsion,
the torsion audit against deleted terms, neutrality per residue.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("data/1JNO.pdb"),
                                reason="1JNO not cached")

mm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402


@pytest.fixture(scope="module")
def built():
    from ala2bg import gramicidin as ga
    top, pos = ga.openmm_model()
    ff = ga.forcefield()
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff)
    X = np.array(pos.value_in_unit(unit.angstrom))
    return ga, top, X, ff, system


def _energy(system, X):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(X * unit.angstrom)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_ga_topology_is_complete_and_in_the_pore_frame(built):
    _, top, X, _, _ = built
    # two chains, each a tree, plus two rings per Trp (4 Trp per chain)
    n_trp = sum(r.name == "TRP" for r in top.residues())
    assert top.getNumBonds() == top.getNumAtoms() - 2 + 2 * n_trp
    z = {(r.chain.id, r.name): X[[a.index for a in r.atoms()], 2].mean()
         for r in top.residues() if r.name in ("FVA", "ETA")}
    assert z[("A", "FVA")] == pytest.approx(0.96, abs=0.1)
    assert z[("A", "ETA")] == pytest.approx(11.57, abs=0.1)
    assert z[("B", "ETA")] == pytest.approx(-11.57, abs=0.1)
    r_xy = np.hypot(X[:, 0], X[:, 1])
    assert np.median(r_xy) < 8.0        # the dimer sits on the z axis


def test_every_residue_is_neutral(built):
    _, top, _, _, system = built
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    for res in top.residues():
        q = sum(nb.getParticleParameters(a.index)[0].value_in_unit(
            unit.elementary_charge) for a in res.atoms())
        assert q == pytest.approx(0.0, abs=1e-6), (res.name, res.index, q)


def test_ga_energy_is_mirror_invariant(built):
    """ff14SB has only phase 0/pi torsions and no CMAP, so D-residues on L
    templates are exact mirror images. A chiral term must break this."""
    _, _, X, _, system = built
    mirror = X * np.array([1.0, 1.0, -1.0])
    assert _energy(system, X) == pytest.approx(_energy(system, mirror),
                                               abs=1e-6)

    import copy
    chiral = copy.deepcopy(system)
    t = mm.PeriodicTorsionForce()
    t.addTorsion(0, 1, 3, 4, 1, np.pi / 3, 10.0)       # a CMAP-like asymmetry
    chiral.addForce(t)
    assert abs(_energy(chiral, X) - _energy(chiral, mirror)) > 1e-3


def test_no_torsion_is_silently_missing(built):
    ga, top, _, ff, system = built
    assert ga.unexplained_missing_torsions(top, system, ff) == []

    # The audit can fail: drop the formyl torsions and it must report them.
    import copy
    broken = copy.deepcopy(system)
    idx = [i for i, f in enumerate(broken.getForces())
           if isinstance(f, mm.PeriodicTorsionForce)][0]
    old = broken.getForce(idx)
    formyl = {a.index for a in top.atoms()
              if a.residue.name == "FVA" and a.name in ("HN", "O1")}
    new = mm.PeriodicTorsionForce()
    for n in range(old.getNumTorsions()):
        p = old.getTorsionParameters(n)
        if not formyl & set(p[:4]):
            new.addTorsion(*p)
    broken.removeForce(idx)
    broken.addForce(new)
    missing = ga.unexplained_missing_torsions(top, broken, ff)
    assert any("FVA:HN" in m or "FVA:O1" in m for m in missing)
