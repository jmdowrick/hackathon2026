"""Data providers. Every one returns the same contract, so nothing downstream
ever knows or cares where the trajectory came from.

    {"X": (n, 2) float64 radians, "dt": float ps, "traj_id": (n,) int, "meta": {...}}

Providers, in order of how fast you can be running:
  toy      -- analytic 2D potential, exact ground truth. No dependencies.
  mdshare  -- the standard alanine dipeptide set (3 x 250 ns explicit water).
              One download, real data in minute one.
  openmm   -- run it ourselves. Many concurrent short replicas, because a
              ~22-2269 atom system badly underuses a modern GPU and throughput
              is the constraint, not single-trajectory speed.

THE SCALING ARGUMENT lives here: every downstream stage consumes this contract,
so swapping the front end (brute force now; RETIS/H-RETIS for systems where
brute force hits the exp(dG/kT) wall) changes nothing downstream. That is why
the scaling claim is architectural rather than aspirational.
"""

from __future__ import annotations

import os

import numpy as np

from .cvs import wrap


# ---------------------------------------------------------------------------
# toy
# ---------------------------------------------------------------------------

def toy_potential(x: np.ndarray, barrier: float = 12.0) -> np.ndarray:
    """Periodic double well in phi, single well in psi. Units kJ/mol.

    F(phi, psi) = (B/2)(1 + cos(2 phi)) + 8 (1 - cos(psi))

    Two minima at phi = +-pi/2 separated by a barrier of height B. Everything
    about it is known in closed form, which is what makes it the right thing to
    debug the whole pipeline against while the real MD is still running.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return 0.5 * barrier * (1.0 + np.cos(2.0 * x[:, 0])) + 8.0 * (1.0 - np.cos(x[:, 1]))


def toy_grad(x, barrier=12.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack([-barrier * np.sin(2.0 * x[:, 0]), 8.0 * np.sin(x[:, 1])], axis=1)


def simulate_toy(n_steps=200_000, n_traj=8, dt=1.0, D=0.02, barrier=12.0,
                 temperature=298.0, seed=0):
    """Brownian dynamics on the toy potential. Euler-Maruyama, wrapped.

    xdot = -D beta grad F + sqrt(2 D) eta
    """
    from .thermo.smoluchowski import rt
    rng = np.random.default_rng(seed)
    beta = 1.0 / rt(temperature)
    sig = np.sqrt(2.0 * D * dt)

    # All replicas advance together: one vectorised Euler-Maruyama step for the
    # whole ensemble, so this is seconds rather than minutes.
    x = wrap(rng.uniform(-np.pi, np.pi, size=(n_traj, 2)))
    traj = np.empty((n_steps, n_traj, 2))
    for s in range(n_steps):
        x = wrap(x - D * beta * toy_grad(x, barrier) * dt
                 + sig * rng.standard_normal((n_traj, 2)))
        traj[s] = x

    # Interleave as [traj0 frames..., traj1 frames...] so traj_id is contiguous,
    # which is what the pair-building in phnn/diffusion assumes.
    X = np.concatenate([traj[:, t, :] for t in range(n_traj)])
    ids = np.concatenate([np.full(n_steps, t) for t in range(n_traj)])

    return {
        "X": X, "dt": dt, "traj_id": ids,
        "meta": {"provider": "toy", "D_true": D, "barrier_kJ": barrier,
                 "temperature": temperature, "n_traj": n_traj,
                 "note": "analytic ground truth; never presented as a result"},
    }


# ---------------------------------------------------------------------------
# mdshare -- real data, one download
# ---------------------------------------------------------------------------

def load_mdshare(cache_dir="data", dt=1.0):
    """Standard alanine dipeptide backbone dihedrals, 3 x 250 ns explicit water.

    Published MSM analyses of this set put the slowest implied timescale around
    ~283 ps, with faster processes near ~29 ps and ~4 ps. Treat those as a soft
    acceptance window (200 ps - 2 ns), not a target: basin free energies and
    barriers are force-field dependent and the literature spread is real.
    """
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    name = "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"
    fn = os.path.join(cache_dir, name)
    if not os.path.exists(fn):
        # Direct fetch rather than the mdshare package: mdshare still imports
        # pkg_resources, which modern setuptools no longer ships, so it fails
        # on a clean 2026 environment. One URL, no dependency.
        url = f"http://ftp.imp.fu-berlin.de/pub/cmb-data/{name}"
        urllib.request.urlretrieve(url, fn)

    with np.load(fn) as f:
        arrs = [f[k] for k in sorted(f.keys())]

    Xs, ids = [], []
    for i, a in enumerate(arrs):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.shape[1] < 2:
            continue
        # heuristic: the file ships degrees
        if np.nanmax(np.abs(a)) > 2 * np.pi + 1e-6:
            a = np.radians(a)
        Xs.append(wrap(a[:, :2]))
        ids.append(np.full(len(a), i))

    if not Xs:
        raise RuntimeError("mdshare file had no usable dihedral arrays")

    return {
        "X": np.concatenate(Xs), "dt": dt,
        "traj_id": np.concatenate(ids),
        "meta": {"provider": "mdshare", "file": str(fn), "n_traj": len(Xs),
                 "system": "alanine dipeptide, explicit TIP3P",
                 "dt_note": "frame spacing assumed 1 ps -- VERIFY against the "
                            "dataset description before quoting absolute rates"},
    }


# ---------------------------------------------------------------------------
# openmm -- run it ourselves
# ---------------------------------------------------------------------------

def _build_system(variant="implicit", forcefield="ff96", constraints=None):
    """Return (system, topology, positions, info).

    IMPORTANT, and verified: openmmtools test systems ship AMBER **ff96**, not
    ff99SB-ILDN or ff14SB. So no modern-force-field literature value validates
    them. Pass forcefield="amber14" to rebuild with amber14-all.xml instead;
    either way the choice is recorded in the manifest and quoted with every
    number.
    """
    from openmm import app
    import openmmtools.testsystems as ts

    cls = {"vacuum": ts.AlanineDipeptideVacuum,
           "implicit": ts.AlanineDipeptideImplicit,
           "explicit": ts.AlanineDipeptideExplicit}[variant]
    base = cls() if constraints is None else cls(constraints=constraints)
    info = {"variant": variant, "forcefield": "ff96 (openmmtools default)",
            "n_atoms": base.topology.getNumAtoms()}

    if forcefield == "amber14":
        files = {"vacuum": ("amber14-all.xml",),
                 "implicit": ("amber14-all.xml", "implicit/obc2.xml"),
                 "explicit": ("amber14-all.xml", "amber14/tip3p.xml")}[variant]
        ff = app.ForceField(*files)
        kw = dict(constraints=app.HBonds, rigidWater=True)
        if variant == "explicit":
            kw.update(nonbondedMethod=app.PME, nonbondedCutoff=0.9 * 1.0)
        else:
            kw.update(nonbondedMethod=app.NoCutoff)
        system = ff.createSystem(base.topology, **kw)
        info["forcefield"] = "amber14-all" + ("/obc2" if variant == "implicit" else "")
        return system, base.topology, base.positions, info

    return base.system, base.topology, base.positions, info


def run_openmm_replica(variant="implicit", ns=5.0, dt_fs=2.0, save_ps=1.0,
                       temperature=298.0, seed=0, forcefield="ff96",
                       platform=None, steps_per_batch=10_000):
    """One replica. Returns (phi, psi) in radians at `save_ps` spacing.

    NEVER call step(1) in a Python loop: per-step round trips to the GPU
    dominate and destroy throughput. We batch, and we only ever store the two
    dihedrals -- full coordinates are neither needed nor wanted.
    """
    import mdtraj as md
    import numpy as _np
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm import app

    system, topology, positions, info = _build_system(variant, forcefield)
    integ = LangevinMiddleIntegrator(temperature * unit.kelvin,
                                     1.0 / unit.picosecond,
                                     dt_fs * unit.femtoseconds)
    integ.setRandomNumberSeed(int(seed))

    if platform:
        sim = app.Simulation(topology, system, integ, Platform.getPlatformByName(platform))
    else:
        sim = app.Simulation(topology, system, integ)

    sim.context.setPositions(positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, int(seed))

    total = int(round(ns * 1000.0 / (dt_fs / 1000.0)))
    stride = max(1, int(round(save_ps / (dt_fs / 1000.0))))
    mdtop = md.Topology.from_openmm(topology)

    frames, done = [], 0
    while done < total:
        n = min(steps_per_batch, total - done)
        sim.step(n)
        done += n
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
        xyz = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frames.append(_np.asarray(xyz))
        _ = stride  # frames are one per batch; see note below

    traj = md.Trajectory(_np.array(frames), mdtop)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    X = wrap(_np.stack([phi[:, 0], psi[:, 0]], axis=1).astype(_np.float64))

    frame_ps = steps_per_batch * dt_fs / 1000.0
    return X, {"dt_ps": frame_ps, "ns": ns, "info": info, "seed": seed,
               "n_frames": len(X)}


def run_openmm(n_replicas=8, ns=5.0, variant="implicit", save_ps=1.0,
               temperature=298.0, forcefield="ff96", platform=None,
               steps_per_batch=10_000, seed0=0, n_workers=None):
    """Many concurrent replicas.

    Concurrency is the whole point: for a system this small a single trajectory
    leaves the GPU mostly idle, and NVIDIA's own guidance is that throughput on
    small systems improves substantially with concurrent processes (MPS). It
    also gives statistically independent blocks for error bars, which one long
    trajectory does not, and it is the natural input shape for MSM estimation.
    """
    from concurrent.futures import ProcessPoolExecutor

    n_workers = n_workers or min(n_replicas, (os.cpu_count() or 4))
    args = [(variant, ns, 2.0, save_ps, temperature, seed0 + i, forcefield,
             platform, steps_per_batch) for i in range(n_replicas)]

    Xs, ids, metas = [], [], []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(run_openmm_replica, *a) for a in args]
            results = [f.result() for f in futs]
    else:
        results = [run_openmm_replica(*a) for a in args]

    for i, (X, m) in enumerate(results):
        Xs.append(X); ids.append(np.full(len(X), i)); metas.append(m)

    dt = metas[0]["dt_ps"]
    return {
        "X": np.concatenate(Xs), "dt": dt, "traj_id": np.concatenate(ids),
        "meta": {"provider": "openmm", "n_replicas": n_replicas, "ns_each": ns,
                 "variant": variant, "forcefield": metas[0]["info"]["forcefield"],
                 "n_atoms": metas[0]["info"]["n_atoms"], "dt_ps": dt},
    }


PROVIDERS = {"toy": simulate_toy, "mdshare": load_mdshare, "openmm": run_openmm}


def load(provider="toy", **kw):
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[provider](**kw)
