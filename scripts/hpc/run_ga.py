"""Gramicidin A: equilibrate, then unbiased production. Standalone.

Depends on NOTHING but openmm and numpy, so the remote machine needs neither
the ala2bg package nor its data. Everything travels in one directory:

    system.xml      serialised System (no barostat, no restraints)
    minimised.pdb   starting coordinates (written by the same Modeller)
    system.json     channel window, build provenance
    run_ga.py       this file

DESIGNED FOR A REMOTE WORKSTATION, NOT A SCHEDULER. The failure modes there
are a dropped ssh session, a reboot, or a busy GPU -- not a wall clock. So:

  * RESUMABLE. Re-running the same command continues from the last
    checkpoint. State is saved as XML (portable across platforms and OpenMM
    builds, unlike binary checkpoints); results are written FIRST and the
    state second, each atomically, and on resume any frames newer than the
    state are dropped, so a crash between the two writes cannot duplicate
    or skip frames.
  * NO SILENT CPU FALLBACK. If the requested platform is missing the run
    refuses to start. (A GPU job quietly running on CPU returns almost
    nothing after days.)
  * --max-hours is checked EVERY FRAME. The KcsA runner checked it only at
    checkpoints, 10 ns apart -- hours of overshoot.

WHAT IS RECORDED, AND IN WHICH FRAME
------------------------------------
The pore frame is RE-MEASURED from the GA backbone every frame (origin at
the dimer junction projected onto the helix axis; +z towards chain A's
C-terminus) -- the same construction as gramicidin.load_gramicidin and
ga_md.measure_frame, which a test checks. Per frame:

  ion_rel   (n_ions, 3)  K+ and Cl- displacement from the pore origin,
                         MINIMUM IMAGE, lab axes. z and r follow from axis.
  axis      (3,)         the pore axis in lab axes
  origin    (3,)         lab coordinates of the pore origin
  box       (3,)         orthorhombic box lengths
  wat_rel   (16, 3)      waters inside the channel cylinder, NaN-padded
  step                   integrator step of the frame

Storing lab-axis displacements rather than (z, r) keeps the choice of
projection with the analysis, where it can be changed and tested. Ions are in
FIXED atom order, so the same arrays serve diffusivity (atom-tracked) and the
permutation-invariant features (which sum over ions).

GA heavy-atom coordinates are kept every --ga-ps for later protein features.

    python run_ga.py --dir . --platform CUDA --ns 100
    python run_ga.py --dir . --platform CUDA --ns 100     # resumes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

T_K = 303.15
GA_RES = {"FVA", "GLY", "ALA", "LEU", "VAL", "TRP", "ETA"}
N_WAT_SLOTS = 16

# (ensemble, dt fs, length ps, k_GA heavy, k_lipid-P z) ; k in kJ/mol/nm^2
EQUIL_STAGES = [
    ("nvt", 1.0, 50.0, 4000.0, 1000.0),
    ("nvt", 1.0, 50.0, 2000.0, 1000.0),
    ("npt", 2.0, 100.0, 1000.0, 400.0),
    ("npt", 2.0, 200.0, 500.0, 200.0),
    ("npt", 4.0, 200.0, 200.0, 40.0),
    ("npt", 4.0, 400.0, 50.0, 0.0),
]
PROD_DT_FS = 4.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# --- indices and the pore frame ------------------------------------------

def indices(top):
    atoms = list(top.atoms())
    ga_chains = {r.chain.index for r in top.residues() if r.name == "FVA"}
    if len(ga_chains) != 2:
        raise ValueError(f"expected two GA chains, found {len(ga_chains)}")
    in_ga = [a for a in atoms if a.residue.chain.index in ga_chains
             and a.residue.name in GA_RES]
    chain_a = min(ga_chains)
    idx = {
        "backbone": [a.index for a in in_ga if a.name in ("N", "CA", "C")],
        "ga_heavy": [a.index for a in in_ga if a.element.symbol != "H"],
        "fva": [[a.index for a in in_ga if a.residue.name == "FVA"
                 and a.residue.chain.index == c] for c in sorted(ga_chains)],
        "eta_a": [a.index for a in in_ga if a.residue.name == "ETA"
                  and a.residue.chain.index == chain_a],
        "ions": [a.index for a in atoms if a.residue.name in ("K", "CL")],
        "ion_names": [a.residue.name for a in atoms
                      if a.residue.name in ("K", "CL")],
        "water_O": [a.index for a in atoms
                    if a.residue.name == "HOH" and a.element.symbol == "O"],
        "lipid_P": [a.index for a in atoms if a.element.symbol == "P"],
    }
    for k in ("backbone", "eta_a", "ions", "water_O", "lipid_P"):
        if not idx[k]:
            raise ValueError(f"no atoms found for {k}")
    return idx


def pore_frame(X, idx):
    """Origin and axis from the GA backbone. X in A, molecules whole."""
    P = X[idx["backbone"]]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    axis = vt[0] / np.linalg.norm(vt[0])
    if np.dot(X[idx["eta_a"]].mean(axis=0) - c, axis) < 0:
        axis = -axis
    junction = 0.5 * (X[idx["fva"][0]].mean(axis=0)
                      + X[idx["fva"][1]].mean(axis=0))
    origin = c + np.dot(junction - c, axis) * axis
    return origin, axis


def min_image(d, box):
    return d - box * np.round(d / box)


def project(rel, axis):
    z = rel @ axis
    r = np.linalg.norm(rel - np.outer(z, axis), axis=1)
    return z, r


def kabsch_rmsd(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    S[-1] *= d
    return float(np.sqrt(max((np.sum(A ** 2) + np.sum(B ** 2)
                              - 2 * S.sum()) / len(A), 0.0)))


# --- systems --------------------------------------------------------------

def make_platform(name, device):
    import openmm as mm
    names = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    if name not in names:
        raise SystemExit(f"FATAL: platform {name!r} not available (have "
                         f"{names}). Refusing to fall back silently.")
    plat = mm.Platform.getPlatformByName(name)
    props = {}
    if device is not None and name in ("CUDA", "OpenCL"):
        props["DeviceIndex"] = str(device)
    return plat, props


def restrained(system, X_nm, idx):
    """Copy of the system with PERIODIC position restraints added.

    periodicdistance, not (x - x0)^2: a non-periodic restraint on an atom
    the context has wrapped across the box gave NaN at step 1 on KcsA
    (bug 6).
    """
    import openmm as mm
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ga = mm.CustomExternalForce(
        "0.5*k_ga*periodicdistance(x,y,z,x0,y0,z0)^2")
    ga.addGlobalParameter("k_ga", 0.0)
    for p in ("x0", "y0", "z0"):
        ga.addPerParticleParameter(p)
    for i in idx["ga_heavy"]:
        ga.addParticle(i, list(X_nm[i]))
    s.addForce(ga)
    lp = mm.CustomExternalForce(
        "0.5*k_p*periodicdistance(x,y,z,x,y,z0)^2")
    lp.addGlobalParameter("k_p", 0.0)
    lp.addPerParticleParameter("z0")
    for i in idx["lipid_P"]:
        lp.addParticle(i, [X_nm[i][2]])
    s.addForce(lp)
    return s


def add_barostat(system):
    import openmm as mm
    from openmm import unit
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    s.addForce(mm.MonteCarloMembraneBarostat(
        1.0 * unit.bar, 0.0 * unit.bar * unit.nanometer, T_K * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFree, 100))
    return s


def new_context(system, dt_fs, platform, props):
    import openmm as mm
    from openmm import unit
    integ = mm.LangevinMiddleIntegrator(T_K * unit.kelvin, 1 / unit.picosecond,
                                        dt_fs * unit.femtosecond)
    return mm.Context(system, integ, platform, props)


def snapshot(ctx):
    from openmm import unit
    st = ctx.getState(getPositions=True, getEnergy=True)
    X = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    v = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    return st, np.asarray(X), np.array([v[0][0], v[1][1], v[2][2]])


def diagnostics(ctx, idx, ch, ref_bb, n_dof):
    from openmm import unit
    st, X, box = snapshot(ctx)
    ke = ctx.getState(getEnergy=True).getKineticEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    T = 2 * ke / (n_dof * 0.0083144626)
    origin, axis = pore_frame(X, idx)
    P = X[idx["lipid_P"]]
    pz = P[:, 2]
    mid = float(np.median(pz))
    up = pz > mid
    rel_w = min_image(X[idx["water_O"]] - origin, box)
    zw, rw = project(rel_w, axis)
    wet = int(np.sum((zw > ch["z_lo"]) & (zw < ch["z_hi"])
                     & (rw < ch["r_max"])))
    rel_i = min_image(X[idx["ions"]] - origin, box)
    zi, ri = project(rel_i, axis)
    ions_in = int(np.sum((zi > ch["z_lo"]) & (zi < ch["z_hi"])
                         & (ri < ch["r_max"])))
    # Area per lipid must exclude the area GA itself occupies in each
    # leaflet; dividing the whole box by the lipid count overstates it by
    # ~10%. GA's footprint: its heavy atoms' 95th-percentile radius from the
    # axis plus a 1.5 A van der Waals margin.
    gz, gr = project(X[idx["ga_heavy"]] - origin, axis)
    ga_area = float(np.pi * (np.percentile(gr, 95) + 1.5) ** 2)
    return {
        "E_pot_kJ": st.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole),
        "T_K": float(T),
        "box_A": box.tolist(),
        "ga_area_A2": ga_area,
        "area_per_lipid_A2": float((box[0] * box[1] - ga_area)
                                   / (len(P) / 2)),
        "PP_thickness_A": float(pz[up].mean() - pz[~up].mean()),
        "tilt_deg": float(np.degrees(np.arccos(min(abs(axis[2]), 1.0)))),
        "junction_offset_A": float(origin[2] - mid),
        "backbone_rmsd_A": kabsch_rmsd(X[idx["backbone"]], ref_bb),
        "waters_in_channel": wet,
        "ions_in_channel": ions_in,
        "finite": bool(np.isfinite(X).all()),
    }


def count_dof(system):
    n = sum(system.getParticleMass(i)._value > 0
            for i in range(system.getNumParticles()))
    return 3 * n - system.getNumConstraints() - 3


EQUIL_CHECKS = {
    # name: (test, explanation)
    "finite": (lambda d: d["finite"], "no NaN coordinates"),
    "temperature": (lambda d: abs(d["T_K"] - T_K) < 0.05 * T_K,
                    "within 5% of target"),
    "area_per_lipid": (lambda d: 54.0 < d["area_per_lipid_A2"] < 68.0,
                       "DMPC ~60 A^2 at 303 K"),
    "PP_thickness": (lambda d: 30.0 < d["PP_thickness_A"] < 42.0,
                     "DMPC ~35 A"),
    "tilt": (lambda d: d["tilt_deg"] < 25.0, "GA roughly along the normal"),
    "junction": (lambda d: abs(d["junction_offset_A"]) < 3.0,
                 "dimer junction near the bilayer centre"),
    "fold": (lambda d: d["backbone_rmsd_A"] < 3.0,
             "beta-helix dimer intact"),
    "wet": (lambda d: d["waters_in_channel"] >= 5,
            "single-file water column present"),
}


# --- stages ---------------------------------------------------------------

def equilibrate(a, system, pdb, idx, ch, platform, props):
    from openmm import unit
    out = os.path.join(a.dir, "equilibrated.xml")
    if os.path.exists(out):
        log("equilibrated.xml exists; skipping equilibration")
        return
    X_nm = np.array(pdb.positions.value_in_unit(unit.nanometer))
    X_A = X_nm * 10.0
    ref_bb = X_A[idx["backbone"]]
    base = restrained(system, X_nm, idx)
    n_dof = count_dof(system)
    report, state = [], None
    for k, (ens, dt, ps, k_ga, k_p) in enumerate(EQUIL_STAGES):
        ps = ps * a.equil_scale
        s = add_barostat(base) if ens == "npt" else base
        ctx = new_context(s, dt, platform, props)
        if state is None:
            ctx.setPositions(pdb.positions)
            ctx.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            ctx.setVelocitiesToTemperature(T_K * unit.kelvin)
        else:
            ctx.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            ctx.setPositions(state.getPositions())
            ctx.setVelocities(state.getVelocities())
        ctx.setParameter("k_ga", k_ga)
        ctx.setParameter("k_p", k_p)
        n = max(1, int(round(ps * 1000.0 / dt)))
        t0 = time.time()
        ctx.getIntegrator().step(n)
        d = diagnostics(ctx, idx, ch, ref_bb, n_dof)
        d.update({"stage": k, "ensemble": ens, "dt_fs": dt, "ps": ps,
                  "k_ga": k_ga, "k_p": k_p,
                  "wall_s": round(time.time() - t0, 1)})
        report.append(d)
        log(f"equil {k} {ens} {ps:6.1f} ps k_ga={k_ga:5.0f}: "
            f"T {d['T_K']:.1f} K, APL {d['area_per_lipid_A2']:.1f}, "
            f"tilt {d['tilt_deg']:.1f}, rmsd {d['backbone_rmsd_A']:.2f}, "
            f"water {d['waters_in_channel']}, ions {d['ions_in_channel']}")
        if not d["finite"]:
            break
        state = ctx.getState(getPositions=True, getVelocities=True,
                             enforcePeriodicBox=False)

    final = report[-1]
    checks = {name: bool(fn(final)) for name, (fn, _) in EQUIL_CHECKS.items()}
    ok = all(checks.values())
    json.dump({"stages": report, "checks": checks,
               "check_meaning": {k: v[1] for k, v in EQUIL_CHECKS.items()},
               "ok": ok, "equil_scale": a.equil_scale},
              open(os.path.join(a.dir, "equil.json"), "w"), indent=1)
    for name, passed in checks.items():
        log(f"  [{'ok' if passed else 'FAIL'}] {name}: "
            f"{EQUIL_CHECKS[name][1]}")
    if not ok and not a.force:
        raise SystemExit("FATAL: equilibration checks failed; see equil.json. "
                         "Not starting production (use --force to override).")
    import openmm as mm
    atomic_write_text(out, mm.XmlSerializer.serialize(state))
    log("wrote equilibrated.xml")


def produce(a, system, pdb, idx, ch, platform, props):
    import openmm as mm
    from openmm import unit
    s = add_barostat(system)
    ctx = new_context(s, PROD_DT_FS, platform, props)
    res_path = os.path.join(a.dir, a.out)
    ckpt_path = os.path.join(a.dir, "production_state.xml")

    stride = max(1, int(round(a.save_ps * 1000.0 / PROD_DT_FS)))
    ga_every = max(1, int(round(a.ga_ps / a.save_ps)))
    total = int(round(a.ns * 1e6 / PROD_DT_FS))

    keys = ("ion_rel", "axis", "origin", "box", "wat_rel", "step")
    buf = {k: [] for k in keys}
    ga_buf, ga_step = [], []
    step0 = None
    if os.path.exists(ckpt_path) and os.path.exists(res_path):
        state = mm.XmlSerializer.deserialize(open(ckpt_path).read())
        ctx.setState(state)
        old = np.load(res_path)
        step0 = int(old["prod_step0"])
        cur = ctx.getStepCount()
        keep = old["step"] <= cur          # frames newer than the state go
        for k in keys:
            buf[k] = list(old[k][keep])
        gk = old["ga_step"] <= cur
        ga_buf, ga_step = list(old["ga_xyz"][gk]), list(old["ga_step"][gk])
        dropped = int((~keep).sum())
        log(f"resuming at step {cur} ({len(buf['step'])} frames"
            + (f"; dropped {dropped} newer than the state" if dropped else "")
            + ")")
    else:
        eq = os.path.join(a.dir, "equilibrated.xml")
        state = mm.XmlSerializer.deserialize(open(eq).read())
        ctx.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        ctx.setPositions(state.getPositions())
        ctx.setVelocities(state.getVelocities())
        ctx.setStepCount(0)
        step0 = 0
        log("starting production from equilibrated.xml")

    def flush(final=False, stopped_early=False):
        cur = ctx.getStepCount()
        atomic_savez(
            res_path,
            ion_rel=np.array(buf["ion_rel"], np.float32).reshape(
                -1, len(idx["ions"]), 3),
            axis=np.array(buf["axis"], np.float64).reshape(-1, 3),
            origin=np.array(buf["origin"], np.float64).reshape(-1, 3),
            box=np.array(buf["box"], np.float64).reshape(-1, 3),
            wat_rel=np.array(buf["wat_rel"], np.float32).reshape(
                -1, N_WAT_SLOTS, 3),
            step=np.array(buf["step"], np.int64),
            ga_xyz=np.array(ga_buf, np.float32).reshape(
                -1, len(idx["ga_heavy"]), 3),
            ga_step=np.array(ga_step, np.int64),
            ion_names=np.array(idx["ion_names"]),
            ion_atoms=np.array(idx["ions"]),
            dt_fs=PROD_DT_FS, save_every_steps=stride, prod_step0=step0,
            channel=json.dumps(ch), complete=bool(final),
            stopped_early=bool(stopped_early),
            ns_requested=a.ns, ns_completed=(cur - step0) * PROD_DT_FS / 1e6)
        # results FIRST, state second: on resume, frames newer than the state
        # are dropped, so a crash between the two writes is harmless.
        st = ctx.getState(getPositions=True, getVelocities=True,
                          getParameters=True, enforcePeriodicBox=False)
        atomic_write_text(ckpt_path, mm.XmlSerializer.serialize(st))

    ckpt_frames = max(1, int(round(a.checkpoint_ps / a.save_ps)))
    t0, frames_since = time.time(), 0
    stopped = False
    while ctx.getStepCount() - step0 < total:
        n = min(stride, total - (ctx.getStepCount() - step0))
        ctx.getIntegrator().step(n)
        _, X, box = snapshot(ctx)
        origin, axis = pore_frame(X, idx)
        rel = min_image(X[idx["ions"]] - origin, box)
        relw = min_image(X[idx["water_O"]] - origin, box)
        zw, rw = project(relw, axis)
        inw = np.where((zw > ch["z_lo"]) & (zw < ch["z_hi"])
                       & (rw < ch["r_max"]))[0]
        inw = inw[np.argsort(zw[inw])][:N_WAT_SLOTS]
        w = np.full((N_WAT_SLOTS, 3), np.nan)
        w[:len(inw)] = relw[inw]
        buf["ion_rel"].append(rel)
        buf["axis"].append(axis)
        buf["origin"].append(origin)
        buf["box"].append(box)
        buf["wat_rel"].append(w)
        buf["step"].append(ctx.getStepCount())
        if len(buf["step"]) % ga_every == 0:
            ga_buf.append(X[idx["ga_heavy"]] - origin)
            ga_step.append(ctx.getStepCount())
        frames_since += 1

        elapsed = time.time() - t0
        if a.max_hours is not None and elapsed > a.max_hours * 3600:
            stopped = True
            log(f"reached --max-hours {a.max_hours}; stopping cleanly")
            break
        if frames_since >= ckpt_frames:
            frames_since = 0
            flush()
            ns = (ctx.getStepCount() - step0) * PROD_DT_FS / 1e6
            log(f"{ns:8.3f} / {a.ns} ns   frames {len(buf['step'])}")

    flush(final=not stopped, stopped_early=stopped)
    log(f"done: {len(buf['step'])} frames -> {res_path}")


def take_lock(d):
    """One run per directory. Two processes writing the same checkpoint
    corrupt it -- easy to do on a workstation after reconnecting and
    forgetting a run is still going. A lock left by a dead process is stale
    and is taken over."""
    path = os.path.join(d, "run.lock")
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        except FileExistsError:
            try:
                pid = int(open(path).read().strip() or 0)
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                log("removing stale lock left by a dead process")
                os.remove(path)
                continue
            except PermissionError:
                pass
            raise SystemExit(f"FATAL: another run (pid {pid}) holds {path}")
    raise SystemExit(f"FATAL: could not take {path}")


def check_atom_order(top, system):
    """The PDB and the System must list the same atoms in the same order.

    A count match is not enough. Hydrogen mass repartitioning moves mass
    from each heavy atom onto its hydrogens, so only heavy atoms with NO
    bonded hydrogen keep their element mass -- ions, phosphates, carbonyl
    oxygens. Those must match exactly; an order mismatch moves them.
    """
    if top.getNumAtoms() != system.getNumParticles():
        raise SystemExit(f"FATAL: PDB has {top.getNumAtoms()} atoms, system "
                         f"{system.getNumParticles()}")
    has_h = set()
    for b in top.bonds():
        for x, y in ((b[0], b[1]), (b[1], b[0])):
            if y.element.symbol == "H":
                has_h.add(x.index)
    n = 0
    for at in top.atoms():
        if at.element.symbol == "H" or at.index in has_h:
            continue
        m = system.getParticleMass(at.index)._value
        # 0.02 amu: force fields round masses (lipid17 N is 14.01, the
        # element 14.0067), while any two elements differ by >= 1 amu.
        if abs(m - at.element.mass._value) > 0.02:
            raise SystemExit(f"FATAL: atom {at.index} ({at.residue.name} "
                             f"{at.name}) has mass {m}, expected "
                             f"{at.element.mass._value}: the PDB and the "
                             "System are not in the same atom order")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=".")
    ap.add_argument("--platform", required=True,
                    help="CUDA, OpenCL or CPU; refused if unavailable")
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index, for a shared workstation")
    ap.add_argument("--ns", type=float, default=100.0, help="production length")
    ap.add_argument("--save-ps", type=float, default=0.4,
                    help="frame spacing; short, for the lag-dependence of D")
    ap.add_argument("--ga-ps", type=float, default=10.0)
    ap.add_argument("--checkpoint-ps", type=float, default=200.0)
    ap.add_argument("--equil-scale", type=float, default=1.0,
                    help="multiply every equilibration stage length")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--out", default="ga_traj.npz")
    ap.add_argument("--force", action="store_true",
                    help="start production even if equilibration checks fail")
    a = ap.parse_args()

    import openmm as mm
    from openmm import app

    system = mm.XmlSerializer.deserialize(
        open(os.path.join(a.dir, "system.xml")).read())
    pdb = app.PDBFile(os.path.join(a.dir, "minimised.pdb"))
    meta = json.load(open(os.path.join(a.dir, "system.json")))
    top = pdb.topology
    check_atom_order(top, system)

    idx = indices(top)
    ch = meta["channel"]
    lock = take_lock(a.dir)
    platform, props = make_platform(a.platform, a.device)
    log(f"openmm {mm.__version__}, platform {a.platform}, "
        f"{top.getNumAtoms()} atoms, {len(idx['ions'])} ions")
    try:
        equilibrate(a, system, pdb, idx, ch, platform, props)
        produce(a, system, pdb, idx, ch, platform, props)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
