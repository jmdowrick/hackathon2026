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

    # Computed inline rather than imported, so this script runs on a cluster
    # that has only openmm and numpy. It previously imported ala2bg, which
    # made the "run preflight on the cluster first" advice impossible to
    # follow.
    def measure_D(Z, R, dt, z_max=9.0, r_max=None):
        z0, z1 = Z[:-1], Z[1:]
        ok = (np.abs(z0) <= z_max) & (np.abs(z1) <= z_max)
        if r_max is not None:
            ok &= (R[:-1] <= r_max) & (R[1:] <= r_max)
        if not ok.any():
            return None, 0
        dz = (z1 - z0)[ok]
        return float(np.mean(dz ** 2) / (2 * dt)), int(ok.sum())

    D_ungated, _ = measure_D(Z, R, dt, r_max=None)
    D_gated, n_pairs = measure_D(Z, R, dt, r_max=6.0)
    gated = {"D_global": D_gated, "n_pairs": n_pairs,
             "available": D_gated is not None}
    ungated = {"D_global": D_ungated, "available": D_ungated is not None}
    check("radial gating changes the answer",
          D_gated is not None and D_ungated is not None
          and abs(D_gated - D_ungated) > 1e-6)
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

    # What matters is that ions are IN the filter, ON the axis and CONFINED
    # -- not that they straddle z = 0, which is only the Thr75 CA plane and
    # an arbitrary landmark. The first version asserted the latter and failed
    # on perfectly good data where all three ions sat above that plane.
    gate = (np.abs(Z) <= 9) & (R <= 6)
    per_ion = gate.sum(axis=0)
    resident = per_ion > 0.5 * len(Z)
    check("ions are resident in the filter",
          int(resident.sum()) >= 2,
          f"{int(resident.sum())} ions gated for >50% of frames")

    if resident.any():
        zf = Z[:, resident][gate[:, resident]]
        rf = R[:, resident][gate[:, resident]]
        check("resident ions are on the pore axis",
              float(np.median(rf)) < 1.5,
              f"median r = {float(np.median(rf)):.2f} A")
        check("resident ions are confined, not diffusing freely",
              float(np.median(Z[:, resident].std(axis=0))) < 1.5,
              f"median per-ion z spread "
              f"{float(np.median(Z[:, resident].std(axis=0))):.2f} A")
        check("filter occupancy spans more than one site",
              float(zf.max() - zf.min()) > 3.0,
              f"gated z span {float(zf.max()-zf.min()):.1f} A "
              f"(site spacing ~3.3 A)")

        # Reported, not asserted: a short equilibration leaves the ions
        # relaxed off the crystallographic sites. Worth watching on the real
        # run, where equilibration is 200 ps rather than a few.
        placed = np.array(meta.get("ion_z_A", []), dtype=float)
        if placed.size:
            final = Z[-1, resident]
            drift = [float(np.min(np.abs(placed - f))) for f in final]
            print(f"     (info) resident ions sit {np.round(drift,2)} A from "
                  f"the nearest placed site")

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
