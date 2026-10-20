"""Build gramicidin A in a DMPC bilayer, and check it before trusting it.

WHAT THIS IS
------------
1JNO -> GA dimer in the pore frame -> DMPC bilayer + TIP3P + KCl (Amber
ff14SB / lipid17; see gramicidin.py for why Amber and how the D-residues and
caps are parameterised) -> minimised. Unlike the KcsA system there IS a
membrane and NO protein restraint: this system is meant for unbiased runs that
train the SRV, and then for weighted ensemble along it.

WHY DMPC. Its hydrophobic thickness matches the GA dimer's length, and it is
the standard host for GA experiments and simulations.

WHAT IS CHECKED, AND WHY
------------------------
addMembrane translates and re-boxes the system. Two KcsA bugs came from
exactly that kind of silent shift (a 5.89 A frame mismatch; 46% of atoms
outside the box), so the pore frame is RE-MEASURED from the built system and
compared, not assumed:

  * GA axis still along z, dimer junction on the bilayer centre
  * every atom inside the box
  * no ion already inside the channel (the starting state is "empty")
  * every residue neutral overall, the system neutral
  * no torsion silently unparameterised
  * the minimised energy is finite and the largest force is moderate

Usage:
    python scripts/ga_md.py --prepare             # build, check, minimise
    python scripts/ga_md.py --prepare --smoke 2   # plus a 2 ps NVT smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = "runs/ga_md"


def check(results, name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'ok' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def measure_frame(top, X):
    """Pore frame from the BUILT coordinates (A): axis, junction, termini."""
    bb = [a.index for a in top.atoms()
          if a.residue.name not in ("HOH", "K", "CL") and
          a.residue.chain.index < 2 and a.name in ("N", "CA", "C")]
    P = X[bb]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    axis = vt[0] / np.linalg.norm(vt[0])
    res = {(r.chain.index, r.name): X[[a.index for a in r.atoms()]].mean(0)
           for r in top.residues() if r.name in ("FVA", "ETA")
           and r.chain.index < 2}
    if np.dot(res[(0, "ETA")] - c, axis) < 0:
        axis = -axis
    junction = 0.5 * (res[(0, "FVA")] + res[(1, "FVA")])
    origin = c + np.dot(junction - c, axis) * axis
    return origin, axis, res


def prepare(padding_nm=1.5, ion_conc=0.15, lipid="DMPC", smoke_ps=0.0,
            platform=None):
    import openmm as mm
    from openmm import app, unit
    from ala2bg import gramicidin as ga
    from ala2bg.features import to_pore_frame

    os.makedirs(OUT, exist_ok=True)
    results = []
    g = ga.load_gramicidin()
    top, pos = ga.openmm_model(g)
    ff = ga.forcefield()

    print(f"Building GA in {lipid}, {padding_nm} nm padding, "
          f"{ion_conc} M KCl ...", flush=True)
    t0 = time.time()
    m = app.Modeller(top, pos)
    m.addMembrane(ff, lipidType=lipid, membraneCenterZ=0 * unit.nanometer,
                  minimumPadding=padding_nm * unit.nanometer,
                  positiveIon="K+", negativeIon="Cl-",
                  ionicStrength=ion_conc * unit.molar)
    top, pos = m.topology, m.positions
    print(f"  {top.getNumAtoms()} atoms in {time.time() - t0:.0f} s")

    X = np.array(pos.value_in_unit(unit.angstrom))
    box = np.array([top.getPeriodicBoxVectors()[i][i].value_in_unit(
        unit.angstrom) for i in range(3)])

    # --- the frame, re-measured -----------------------------------------
    origin, axis, termini = measure_frame(top, X)
    tilt = float(np.degrees(np.arccos(min(abs(axis[2]), 1.0))))
    check(results, "GA axis along z after building", tilt < 2.0,
          f"tilt {tilt:.2f} deg")
    p_atoms = [a.index for a in top.atoms()
               if a.residue.name.startswith(("DMP", "PC", "PA", "PE", "PO"))
               and a.element is not None and a.element.symbol == "P"]
    mem_z = float(X[p_atoms, 2].mean()) if p_atoms else float("nan")
    check(results, "dimer junction on the bilayer centre",
          np.isfinite(mem_z) and abs(origin[2] - mem_z) < 1.5,
          f"junction z {origin[2]:.2f} A, phosphate mean z {mem_z:.2f} A "
          f"({len(p_atoms)} P atoms)")
    if p_atoms:
        up = X[p_atoms, 2] > mem_z
        thick = float(X[p_atoms, 2][up].mean() - X[p_atoms, 2][~up].mean())
        check(results, "bilayer P-P thickness plausible for DMPC",
              30.0 < thick < 42.0, f"{thick:.1f} A")

    # Molecule CENTROIDS, not atoms: whole lipids straddle the periodic
    # boundary by design (up to ~30 A long), so atom extent overshoots the
    # box legitimately. Bug 7's signature was different -- a frame shift
    # that put 46% of the system outside.
    cents = np.array([X[[a.index for a in r.atoms()]].mean(axis=0)
                      for r in top.residues()])
    centre = 0.5 * (cents.min(axis=0) + cents.max(axis=0))
    outside = np.any(np.abs(cents - centre) > box / 2 + 2.0, axis=1)
    check(results, "molecule centroids inside the box", outside.mean() < 0.01,
          f"{100 * outside.mean():.2f}% outside; centroid span "
          f"{np.round(np.ptp(cents, axis=0), 1).tolist()} vs box "
          f"{np.round(box, 1).tolist()} A")

    # --- ions -------------------------------------------------------------
    ch = ga.channel_from_structure(g)
    ions = [a.index for a in top.atoms() if a.residue.name in ("K", "CL")]
    zi, ri = to_pore_frame(X[ions][None], origin, axis, box=box)
    from ala2bg.features import in_channel
    n_in = float(in_channel(zi, ri, ch, soft=False).sum())
    check(results, "no ion starts inside the channel", n_in == 0,
          f"{len(ions)} ions, {int(n_in)} in channel")

    # --- the system ---------------------------------------------------------
    system = ff.createSystem(top, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, hydrogenMass=4 * unit.amu)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(
        unit.elementary_charge) for i in range(system.getNumParticles())])
    # lipid17 DMPC charges round to +1e-6 e per lipid; ~1e-4 e in total is
    # that, not a missing ion. Per-residue neutrality below is the real test.
    check(results, "system neutral", abs(q.sum()) < 1e-3,
          f"net {q.sum():+.2e} e")
    bad_res = []
    for r in top.residues():
        if r.name in ("K", "CL"):
            continue
        qr = q[[a.index for a in r.atoms()]].sum()
        if abs(qr) > 1e-5:
            bad_res.append(f"{r.name}{r.index}:{qr:+.3f}")
    check(results, "every non-ion residue neutral", not bad_res,
          ", ".join(bad_res[:5]))
    gsys = ff.createSystem(ga.openmm_model(g)[0],
                           nonbondedMethod=app.NoCutoff)
    missing = ga.unexplained_missing_torsions(ga.openmm_model(g)[0], gsys, ff)
    check(results, "no GA torsion silently unparameterised", not missing,
          ", ".join(missing[:3]))

    # --- minimise -------------------------------------------------------
    integ = mm.LangevinMiddleIntegrator(303.15 * unit.kelvin, 1 / unit.picosecond,
                                        0.004 * unit.picoseconds)
    plat = mm.Platform.getPlatformByName(platform) if platform else None
    sim = (app.Simulation(top, system, integ, plat) if plat
           else app.Simulation(top, system, integ))
    sim.context.setPositions(pos)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print("  minimising ...", flush=True)
    sim.minimizeEnergy(maxIterations=5000)
    st = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = np.linalg.norm(st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer), axis=1)
    check(results, "minimised energy finite and lower",
          np.isfinite(e1) and e1 < e0.value_in_unit(unit.kilojoule_per_mole),
          f"{e0.value_in_unit(unit.kilojoule_per_mole):.3e} -> {e1:.3e} kJ/mol")
    check(results, "largest force after minimisation moderate",
          np.isfinite(F).all() and F.max() < 5e3,
          f"max |F| {F.max():.0f} kJ/mol/nm on atom {int(F.argmax())}")

    Xm = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    origin_m, axis_m, _ = measure_frame(top, np.asarray(Xm))

    if smoke_ps > 0:
        sim.context.setVelocitiesToTemperature(303.15 * unit.kelvin)
        n = int(round(smoke_ps / 0.004))
        print(f"  smoke run {smoke_ps} ps ({n} steps) ...", flush=True)
        t0 = time.time()
        sim.step(n)
        s2 = sim.context.getState(getEnergy=True, getPositions=True)
        e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        X2 = s2.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        check(results, "smoke run stays finite",
              np.isfinite(e2) and np.isfinite(X2).all(),
              f"E {e2:.3e} kJ/mol after {smoke_ps} ps "
              f"({(time.time() - t0) / max(smoke_ps, 1e-9):.1f} s/ps)")

    # --- write ---------------------------------------------------------
    with open(f"{OUT}/system.xml", "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    with open(f"{OUT}/topology.pkl", "wb") as f:
        pickle.dump(top, f)
    with open(f"{OUT}/minimised.pdb", "w") as f:
        app.PDBFile.writeFile(top, st.getPositions(), f, keepIds=True)
    meta = {
        "system": "gramicidin A (1JNO) in DMPC, Amber ff14SB/lipid17, TIP3P",
        "n_atoms": top.getNumAtoms(), "lipid": lipid,
        "padding_nm": padding_nm, "KCl_M": ion_conc,
        "box_A": box.tolist(),
        "pore_origin_A": origin_m.tolist(), "pore_axis": axis_m.tolist(),
        "channel": {"z_lo": ch.z_lo, "z_hi": ch.z_hi, "r_max": ch.r_max},
        "ion_atoms": ions,
        "frame_note": ("origin/axis re-measured from the MINIMISED "
                       "coordinates; use these, not load_gramicidin()'s, "
                       "for trajectory analysis"),
        "cap_charges": "by analogy (ACE / NME + Ser), not RESP",
        "hydrogen_mass_amu": 4, "timestep_fs": 4,
        "checks": results,
    }
    with open(f"{OUT}/system.json", "w") as f:
        json.dump(meta, f, indent=1)
    n_fail = sum(not r["ok"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed; "
          f"wrote {OUT}/system.xml, topology.pkl, minimised.pdb, system.json")
    return n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--padding", type=float, default=1.5, help="nm")
    ap.add_argument("--conc", type=float, default=0.15, help="KCl, M")
    ap.add_argument("--lipid", default="DMPC")
    ap.add_argument("--smoke", type=float, default=0.0, help="ps")
    ap.add_argument("--platform", default=None)
    a = ap.parse_args()
    if a.prepare:
        return 1 if prepare(a.padding, a.conc, a.lipid, a.smoke,
                            a.platform) else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
