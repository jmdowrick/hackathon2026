"""Pre-flight checks for the HPC bundle. Run BEFORE submitting.

A cluster job that fails after six hours, or worse succeeds and returns
plausible-looking nonsense, is expensive. These checks run in a couple of
minutes locally and are meant to catch the second case as well as the first.

    python scripts/hpc/preflight.py --dir runs/kcsa_md/bundle

Each check prints PASS or FAIL with the number it checked, so a failure says
what is wrong rather than just that something is.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""),
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/kcsa_md/bundle")
    ap.add_argument("--ns", type=float, default=0.08,
                    help="length of the live test run")
    ap.add_argument("--platform", default=None)
    a = ap.parse_args()
    d = a.dir

    from openmm import app, unit, XmlSerializer, NonbondedForce, Platform

    print("\n1. BUNDLE CONTENTS")
    needed = ["system.xml", "topology.pkl", "minimised.pdb", "system.json",
              "run_kcsa.py"]
    for f in needed:
        check(f"{f} present", os.path.exists(os.path.join(d, f)))

    print("\n2. SYSTEM INTEGRITY")
    system = XmlSerializer.deserialize(open(os.path.join(d, "system.xml")).read())
    topology = pickle.load(open(os.path.join(d, "topology.pkl"), "rb"))
    pdb = app.PDBFile(os.path.join(d, "minimised.pdb"))
    meta = json.load(open(os.path.join(d, "system.json")))

    n_sys = system.getNumParticles()
    n_top = topology.getNumAtoms()
    n_pdb = len(pdb.positions)
    check("system / topology / pdb atom counts agree",
          n_sys == n_top == n_pdb, f"{n_sys} / {n_top} / {n_pdb}")

    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), NonbondedForce)][0]
    q = sum(nb.getParticleParameters(i)[0]._value for i in range(nb.getNumParticles()))
    check("system is neutral", abs(q) < 1e-6, f"net charge {q:+.2e} e")
    check("PME electrostatics", nb.getNonbondedMethod() == NonbondedForce.PME)

    forces = [system.getForce(i).__class__.__name__
              for i in range(system.getNumForces())]
    check("position restraints present", "CustomExternalForce" in forces)

    # the restraint must be periodic-aware, or atoms wrapping the box explode
    ext = [system.getForce(i) for i in range(system.getNumForces())
           if system.getForce(i).__class__.__name__ == "CustomExternalForce"]
    if ext:
        expr = ext[0].getEnergyFunction()
        check("restraint uses periodicdistance", "periodicdistance" in expr,
              expr[:46])
        check("restrained atom count is sensible",
              100 < ext[0].getNumParticles() < n_sys // 4,
              f"{ext[0].getNumParticles()} atoms")

    x = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    box = np.array(topology.getUnitCellDimensions().value_in_unit(unit.angstrom))
    is_prot = np.array([at.residue.name not in ("HOH", "K", "CL")
                        for at in topology.atoms()])

    # The SOLUTE must fit the box. Water need not: OpenMM does not wrap
    # coordinates when writing a PDB, so solvent that diffused across the
    # boundary during equilibration legitimately appears a few angstroms
    # outside, and PBC is handled internally regardless.
    #
    # The failure this is really guarding against is the frame mismatch that
    # produced the first NaN here: crystal coordinates far from the origin
    # gave a span of 164 A against an 86 A box -- 90% excess, WITH protein
    # atoms outside. Testing the total span alone flags harmless unwrapped
    # water (8% excess, no protein) as a failure, so it tests the solute.
    prot_span = x[is_prot].max(0) - x[is_prot].min(0)
    check("solute fits inside the periodic box",
          bool(np.all(prot_span <= box)),
          f"protein span {np.round(prot_span,1)} vs box {np.round(box,1)}")

    centre = x.mean(0)
    outside = ((x < centre - box / 2) | (x > centre + box / 2)).any(1)
    check("no solute atom outside the box",
          int((outside & is_prot).sum()) == 0,
          f"{int((outside & is_prot).sum())} protein atoms outside")
    check("solvent overflow is small",
          float(outside.mean()) < 0.15,
          f"{100*outside.mean():.1f}% of atoms outside (unwrapped water)")
    check("no NaN in starting coordinates", not bool(np.isnan(x).any()))

    print("\n3. ATOM SELECTION")
    ions = [at.index for at in topology.atoms()
            if at.element is not None and at.element.symbol == "K"
            and at.residue.name in ("K", "K+")]
    ring = [at.index for at in topology.atoms()
            if at.name == "CA" and at.residue.id == "75"]
    check("K+ ions found", len(ions) > 0, f"{len(ions)} ions")
    check("filter ring has 4 subunits", len(ring) == 4, f"{len(ring)} Thr75 CA")

    o, ax = np.array(meta["origin"]), np.array(meta["axis"])
    rel = x[ring] - x[ring].mean(0)
    _, _, vt = np.linalg.svd(rel)
    axis_live = vt[-1] * np.sign(vt[-1][2])
    check("stored axis matches the live filter ring",
          abs(abs(float(np.dot(axis_live, ax / np.linalg.norm(ax)))) - 1) < 0.05,
          f"dot {float(np.dot(axis_live, ax/np.linalg.norm(ax))):.4f}")

    # the three placed ions should still be near their crystallographic sites
    oo = x[ring].mean(0)
    zi = (x[ions] - oo) @ axis_live
    ri = np.linalg.norm((x[ions] - oo) - np.outer(zi, axis_live), axis=1)
    in_filter = (np.abs(zi) <= 9) & (ri <= 6)
    check("placed ions are on-axis inside the filter",
          int(in_filter.sum()) >= 2,
          f"{int(in_filter.sum())} of {len(ions)} within z<=9 A and r<=6 A")

    print("\n4. LIVE SHORT RUN")
    plats = [Platform.getPlatform(i).getName()
             for i in range(Platform.getNumPlatforms())]
    print(f"     platforms available here: {plats}")
    import subprocess
    cmd = [sys.executable, os.path.join(d, "run_kcsa.py"), "--dir", d,
           "--ns", str(a.ns), "--equil-ps", "4", "--checkpoint-ns",
           str(a.ns / 2), "--out", "preflight.npz"]
    if a.platform:
        cmd += ["--platform", a.platform]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    check("short run completes", r.returncode == 0,
          (r.stderr.strip().splitlines() or ["ok"])[-1][:80])
    if r.returncode != 0:
        return summary()

    print("\n5. OUTPUT SANITY")
    out = np.load(os.path.join(d, "preflight.npz"), allow_pickle=True)
    check("Z and R both saved", "Z" in out and "R" in out,
          f"keys {sorted(out.files)}")
    Z, R, dt = out["Z"], out["R"], float(out["dt_ps"])
    check("frame spacing is sub-picosecond", dt <= 1.0, f"dt = {dt} ps")
    check("no NaN in recorded positions",
          not (np.isnan(Z).any() or np.isnan(R).any()))
    check("checkpointing wrote a partial flag", "partial" in out.files,
          f"partial={bool(out['partial'])}")

    print("\n6. PHYSICS SANITY")
    from ala2bg import kcsa
    ungated = kcsa.measure_filter_D(Z, dt)
    gated = kcsa.measure_filter_D(Z, dt, r_traj=R, r_max=6.0)
    check("radial gating is actually applied", gated.get("radially_gated") is True)
    if ungated.get("available") and gated.get("available"):
        check("radial gating lowers D",
              gated["D_global"] < ungated["D_global"],
              f"{ungated['D_global']:.4f} -> {gated['D_global']:.4f} A^2/ps")
    if gated.get("available"):
        D = gated["D_global"]
        check("D is in a physical range for a filter ion",
              0.001 < D < 0.5,
              f"D = {D:.4f} A^2/ps (bulk K+ ~0.2; confined filter ion lower)")
        check("enough displacement pairs to mean anything",
              gated["n_pairs"] > 200, f"{gated['n_pairs']} pairs")

    zf = Z[(np.abs(Z) <= 9) & (R <= 6)]
    if zf.size:
        check("ions sample the filter region",
              float(zf.min()) < 0 < float(zf.max()),
              f"z range {zf.min():.1f} .. {zf.max():.1f} A")

    return summary()


def summary():
    bad = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("\nFAILED:")
        for n in bad:
            print(f"  * {n}")
        print("\nDo not submit until these are resolved.")
        return 1
    print("\nBundle looks sound. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
