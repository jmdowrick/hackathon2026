"""Standalone KcsA production run for an HPC node. No project imports.

Deliberately depends on NOTHING but openmm and numpy, so the cluster job does
not need the ala2bg package, its environment, or any of its data files. The
system has already been built, solvated, neutralised, minimised and warmed up
locally; this script only integrates and records ion positions.

Everything it needs travels in one directory:

    system.xml       serialised OpenMM System (solvated, neutral, restrained)
    topology.pkl     the matching Topology
    minimised.pdb    equilibrated starting coordinates
    system.json      pore axis/origin and the ion placement

It writes filter_ions.npz, which is ~1 MB for a 100 ns run because only the
ion coordinates along the pore axis are stored, not trajectories. Copy that
one file back and analyse locally.

    python run_kcsa.py --ns 100 --platform CUDA --out filter_ions.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

DT_PS = 0.004          # 4 fs, with hydrogen mass repartitioning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding system.xml etc")
    ap.add_argument("--ns", type=float, default=100.0)
    ap.add_argument("--save-ps", type=float, default=0.4,
                    help="frame spacing; must sit inside the diffusive regime")
    ap.add_argument("--equil-ps", type=float, default=200.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--out", default="filter_ions.npz")
    ap.add_argument("--checkpoint-ns", type=float, default=10.0,
                    help="write partial results this often, so a job that "
                         "hits its wall-clock limit still returns data")
    a = ap.parse_args()

    from openmm import app, unit, XmlSerializer, LangevinMiddleIntegrator, Platform

    d = a.dir
    system = XmlSerializer.deserialize(open(os.path.join(d, "system.xml")).read())
    topology = pickle.load(open(os.path.join(d, "topology.pkl"), "rb"))
    pdb = app.PDBFile(os.path.join(d, "minimised.pdb"))
    meta = json.load(open(os.path.join(d, "system.json")))

    integrator = LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond,
                                          DT_PS * unit.picosecond)
    kw = {}
    if a.platform:
        try:
            kw["platform"] = Platform.getPlatformByName(a.platform)
        except Exception:
            print(f"platform {a.platform} unavailable; using the default")
    sim = app.Simulation(topology, system, integrator, **kw)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(298 * unit.kelvin)
    print("platform:", sim.context.getPlatform().getName(), flush=True)

    sim.step(int(a.equil_ps / DT_PS))

    ion_idx = [at.index for at in topology.atoms()
               if at.element is not None and at.element.symbol == "K"
               and at.residue.name in ("K", "K+")]
    ring_idx = [at.index for at in topology.atoms()
                if at.name == "CA" and at.residue.id == "75"]
    print(f"tracking {len(ion_idx)} K+; filter ring atoms: {len(ring_idx)}",
          flush=True)

    stride = max(1, int(round(a.save_ps / DT_PS)))
    total = int(a.ns * 1000 / DT_PS)
    ckpt = max(1, int(a.checkpoint_ns * 1000 / DT_PS))

    Z, R, done, t0 = [], [], 0, time.time()
    o0, ax0 = np.array(meta["origin"]), np.array(meta["axis"])

    def flush(partial):
        arr = np.array(Z)
        np.savez_compressed(
            os.path.join(d, a.out), Z=arr, R=np.array(R), dt_ps=stride * DT_PS,
            ion_names=meta.get("ion_names", []), partial=partial,
            ns_completed=done * DT_PS / 1000.0)

    while done < total:
        n = min(stride, total - done)
        sim.step(n)
        done += n
        st = sim.context.getState(getPositions=True)
        x = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if len(ring_idx) >= 3:
            ring = x[ring_idx]
            o = ring.mean(axis=0)
            _, _, vt = np.linalg.svd(ring - o)
            ax = vt[-1]
            if ax[2] < 0:
                ax = -ax
        else:
            o, ax = o0, ax0
        rel = x[ion_idx] - o
        zz = rel @ ax
        Z.append(zz)
        # radial distance from the pore axis. REQUIRED, not diagnostic: the
        # axial coordinate alone cannot distinguish an ion in the filter from
        # a bulk ion 30 A off-axis at the same height, and counting the latter
        # inflates D by more than an order of magnitude.
        R.append(np.linalg.norm(rel - np.outer(zz, ax), axis=1))

        if done % ckpt < stride:
            ns = done * DT_PS / 1000.0
            rate = ns / max(time.time() - t0, 1e-9) * 86400
            print(f"  {ns:7.2f} ns   {rate:7.1f} ns/day", flush=True)
            flush(True)

    flush(False)
    ns = done * DT_PS / 1000.0
    print(f"done: {ns:.2f} ns, {len(Z)} frames at {stride*DT_PS} ps -> {a.out}")


if __name__ == "__main__":
    main()
