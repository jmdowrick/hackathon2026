"""The standalone GA runner and the loader that reads its output.

The runner cannot import ala2bg, so its pore frame is a second implementation;
the first test pins the two together. The slow test runs the shipped bundle
end to end (scripts/ga_md.py preflight).
"""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILT = os.path.join(ROOT, "runs", "ga_md", "minimised.pdb")
needs_build = pytest.mark.skipif(not os.path.exists(BUILT),
                                 reason="run scripts/ga_md.py --prepare first")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load("run_ga", "scripts/hpc/run_ga.py")


@pytest.fixture(scope="module")
def built():
    from openmm import app, unit
    import openmm as mm
    pdb = app.PDBFile(BUILT)
    system = mm.XmlSerializer.deserialize(
        open(os.path.join(ROOT, "runs", "ga_md", "system.xml")).read())
    X = np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    return pdb.topology, system, X


@needs_build
def test_runner_frame_matches_the_package_frame(runner, built):
    """Two implementations of one frame (bug 3 was a 5.89 A disagreement)."""
    top, _, X = built
    ga_md = _load("ga_md", "scripts/ga_md.py")
    o1, a1 = runner.pore_frame(X, runner.indices(top))
    o2, a2, _ = ga_md.measure_frame(top, X)
    assert np.linalg.norm(o1 - o2) < 1e-6
    assert np.dot(a1, a2) > 1 - 1e-9

    # And the frame is the NMR structure's: chain A's backbone centroid sits
    # at the same z (0.36 A of minimisation shift). The flexible ethanolamine
    # moves 1.2 A on minimising, so it is not a frame reference.
    from ala2bg import gramicidin as ga
    from openmm import unit
    idx = runner.indices(top)
    atoms = list(top.atoms())
    bb_a = [i for i in idx["backbone"] if atoms[i].residue.chain.index == 0]
    z_built = (X[bb_a].mean(axis=0) - o1) @ a1
    t0, p0 = ga.openmm_model()
    Y = np.array(p0.value_in_unit(unit.angstrom))
    bb0 = [a.index for a in t0.atoms()
           if a.residue.chain.index == 0 and a.name in ("N", "CA", "C")]
    assert z_built == pytest.approx(Y[bb0].mean(axis=0)[2], abs=0.6)


@needs_build
def test_atom_order_check_catches_a_reordering(runner, built):
    import copy
    top, system, _ = built
    assert runner.check_atom_order(top, system) > 1000
    bad = copy.deepcopy(system)
    ions = [a.index for a in top.atoms() if a.residue.name == "K"]
    p = [a.index for a in top.atoms() if a.element.symbol == "P"]
    mi, mp = bad.getParticleMass(ions[0]), bad.getParticleMass(p[0])
    bad.setParticleMass(ions[0], mp)
    bad.setParticleMass(p[0], mi)
    with pytest.raises(SystemExit, match="atom order"):
        runner.check_atom_order(top, bad)


def test_min_image_and_projection(runner):
    box = np.array([60.0, 60.0, 80.0])
    d = runner.min_image(np.array([[0.5, 0.0, 79.0]]), box)
    assert d[0] == pytest.approx([0.5, 0.0, -1.0])
    z, r = runner.project(d, np.array([0.0, 0.0, 1.0]))
    assert z[0] == pytest.approx(-1.0) and r[0] == pytest.approx(0.5)


def test_lock_refuses_a_live_holder_and_takes_a_stale_one(runner, tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="another run"):
        runner.take_lock(str(tmp_path))
    lock.write_text("999999")
    assert runner.take_lock(str(tmp_path)) == str(lock)
    assert lock.read_text() == str(os.getpid())


def _fake_run(path, steps, drop=()):
    from ala2bg.gramicidin import RUN_KEYS
    n, k = len(steps), 3
    arrays = dict(
        ion_rel=np.zeros((n, k, 3), np.float32),
        axis=np.tile([0.0, 0.0, 1.0], (n, 1)), origin=np.zeros((n, 3)),
        box=np.full((n, 3), 60.0), wat_rel=np.full((n, 16, 3), np.nan),
        step=np.array(steps), ion_names=np.array(["K", "K", "CL"]),
        dt_fs=4.0, save_every_steps=100, prod_step0=0,
        channel='{"z_lo": -13, "z_hi": 13, "r_max": 4}', complete=True)
    arrays["ion_rel"][:, 0] = [0.0, 3.0, 4.0]       # K on z=4, r=3
    for key in drop:
        arrays.pop(key)
    assert set(RUN_KEYS) - set(drop) <= set(arrays)
    np.savez(path, **arrays)


def test_load_run_projects_and_selects_ions(tmp_path):
    from ala2bg import gramicidin as ga
    p = tmp_path / "t.npz"
    _fake_run(p, [100, 200, 300])
    r = ga.load_run(p)
    assert r["z"].shape == (3, 2)                      # the two K, not Cl
    assert r["z"][0, 0] == pytest.approx(4.0)
    assert r["r"][0, 0] == pytest.approx(3.0)
    assert r["dt_ps"] == pytest.approx(0.4)


def test_load_run_refuses_missing_arrays_and_gaps(tmp_path):
    from ala2bg import gramicidin as ga
    p = tmp_path / "old.npz"
    _fake_run(p, [100, 200, 300], drop=("wat_rel",))
    with pytest.raises(ValueError, match="lacks"):
        ga.load_run(p)
    q = tmp_path / "gap.npz"
    _fake_run(q, [100, 200, 400])
    with pytest.raises(ValueError, match="gaps or duplicates"):
        ga.load_run(q)


@pytest.mark.slow
@needs_build
def test_preflight_runs_the_shipped_bundle_end_to_end():
    import openmm as mm
    names = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    plat = next((p for p in ("CUDA", "OpenCL", "CPU") if p in names))
    ga_md = _load("ga_md", "scripts/ga_md.py")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        results, _ = ga_md.preflight(plat)
    finally:
        os.chdir(cwd)
    failed = [r for r in results if not r["ok"]]
    assert not failed, failed
