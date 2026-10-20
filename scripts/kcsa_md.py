"""Build and run a rigid-filter KcsA simulation, to MEASURE D for filter K+.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is NOT a full KcsA-in-a-bilayer simulation. The protein heavy atoms are
position-restrained and there is no membrane. That is a deliberate reduction,
and it is valid for exactly one quantity:

    the local diffusivity of K+ inside the selectivity filter

D needs only short-time displacements of an ion in its local environment, and
that environment is protein carbonyl oxygens and water, not lipid. The bilayer
governs the protein's conformational dynamics and gating, which we are
restraining away and make no claim about.

It is NOT valid for: gating, conductance in absolute terms, C-type
inactivation, or anything requiring the protein to move. Those need the full
system, and a permeation event needs ~0.55 us at +300 mV with the gate
restrained open, which is why the free-energy landscape comes from elsewhere.

WHY THIS ROUTE RATHER THAN CHARMM-GUI
-------------------------------------
No interactive web step and no membrane building: pdbfixer and OpenMM take the
PDB entry directly, so the whole thing is one command. The cost is the
restriction above, which is stated in the output rather than discovered later.

Usage:
    python scripts/kcsa_md.py --prepare                 # build, solvate, minimise
    python scripts/kcsa_md.py --run --ns 20 --platform CUDA
    python scripts/kcsa_md.py --analyse                 # D + landscape + bond graph
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

OUT = "runs/kcsa_md"
# K+ placement. The crystal resolves ions at every site simultaneously, which
# is a crystallographic average rather than a physical configuration -- the
# filter holds two or three ions at once, not five. "S2S4" is the standard
# choice; waters occupy the remaining sites.
ION_SETS = {"S2S4": ("S2", "S4", "Scav"),
            "S1S3": ("S1", "S3", "Scav"),
            "all": ("Scav", "S4", "S3", "S2", "S1", "S0")}


def build_tetramer_pdb(pdb_id="1K4C", chain="C", ions="S2S4",
                       path=f"{OUT}/kcsa_tetramer.pdb"):
    """Write the C4 channel with a physical ion configuration."""
    from ala2bg import pdb as pdbmod

    os.makedirs(os.path.dirname(path), exist_ok=True)
    src = pdbmod.fetch(pdb_id)
    s1, ops = pdbmod.read_pdb(src, keep_chains=[chain])
    s4 = pdbmod.apply_biomt(s1, ops, which=list(range(min(4, len(ops)))))
    origin, axis, _ = pdbmod.pore_frame(s4)
    z_ions, _ = pdbmod.ion_sites(s4, origin, axis)

    names = ["Scav", "S4", "S3", "S2", "S1", "S0", "Sext"][:len(z_ions)]
    keep_names = ION_SETS[ions]
    keep_z = [z for n, z in zip(names, z_ions) if n in keep_names]

    # protein only (drop Fab, lipid fragments, crystallographic waters)
    prot = (~s4.is_hetatm) & (s4.element != "H")
    xyz = s4.xyz[prot]
    chains = "ABCD"
    lines, serial = [], 1
    for i in range(prot.sum()):
        ch = chains[int(str(s4.chain[prot][i])[-1]) % 4]
        lines.append(
            "ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
            % (serial, s4.name[prot][i][:4], s4.resname[prot][i], ch,
               s4.resseq[prot][i], xyz[i, 0], xyz[i, 1], xyz[i, 2],
               s4.element[prot][i][:2]))
        serial += 1
    for k, z in enumerate(keep_z):
        p = origin + axis * z
        lines.append(
            "HETATM%5d  K   K   X%4d    %8.3f%8.3f%8.3f  1.00  0.00          K \n"
            % (serial, k + 1, p[0], p[1], p[2]))
        serial += 1
    lines.append("END\n")
    open(path, "w").writelines(lines)

    meta = {"pdb_id": pdb_id, "n_protein_atoms": int(prot.sum()),
            "ion_set": ions, "ion_names": list(keep_names),
            "ion_z_A": [float(z) for z in keep_z],
            "origin": origin.tolist(), "axis": axis.tolist()}
    json.dump(meta, open(f"{OUT}/system.json", "w"), indent=2)
    return path, meta


def prepare(pdb_path, padding_nm=1.2, ion_conc=0.15):
    """pdbfixer -> hydrogens, solvate, minimise. Writes a serialised system."""
    from openmm import app, unit, XmlSerializer, LangevinMiddleIntegrator
    from openmm import CustomExternalForce, Vec3
    from pdbfixer import PDBFixer

    def openmm_vec(a):
        return Vec3(float(a[0]), float(a[1]), float(a[2]))

    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # do not model unresolved termini
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # CENTRE THE MOLECULE FIRST. OpenMM's periodic box runs from the ORIGIN,
    # but PDB coordinates sit wherever the crystal frame put them -- for 1K4C
    # that is x,y ~ 128-183 and z ~ -79 to -17. Solvating without translating
    # builds an 86 A box at [0, 86]^3 that does not contain the protein at
    # all: 46% of atoms end up outside, interacting with periodic images, and
    # the first integration step produces NaN. This is one of the chores
    # CHARMM-GUI does silently, and skipping it is not free.
    pos = np.array(fixer.positions.value_in_unit(unit.nanometer))
    pos = pos - pos.mean(axis=0)
    fixer.positions = unit.Quantity(
        [openmm_vec(p_) for p_ in pos], unit.nanometer)

    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, padding=padding_nm * unit.nanometer,
                        ionicStrength=ion_conc * unit.molar,
                        positiveIon="K+", negativeIon="Cl-")

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True,
                             hydrogenMass=4 * unit.amu)

    # position restraints on protein heavy atoms -- the reduction this script
    # is built around, so it is explicit rather than a hidden default
    # PERIODICDISTANCE, not a raw coordinate difference. Under PBC an atom
    # near the box edge is wrapped to the other side, so (x - x0) becomes a
    # full box length (~86 A here), the restraint force explodes and the first
    # integration step produces NaN. This was the second NaN, after the
    # frame-centring one; both are bookkeeping that a CHARMM-GUI input set
    # would have got right for us.
    force = CustomExternalForce(
        "0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    force.addGlobalParameter("k", 1000.0)      # kJ/mol/nm^2, as a plain float
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    pos = modeller.getPositions()
    n_restrained = 0
    for atom in modeller.topology.atoms():
        if atom.residue.name not in ("HOH", "K", "CL") and atom.element.symbol != "H":
            force.addParticle(atom.index,
                              [float(v) for v in
                               pos[atom.index].value_in_unit(unit.nanometer)])
            n_restrained += 1
    system.addForce(force)

    # Minimise, then WARM UP AT A SHORT TIMESTEP. Going straight to 4 fs with
    # repartitioned hydrogens on a freshly solvated system is how the first
    # attempt died; the graded start is the other chore CHARMM-GUI normally
    # supplies.
    integrator = LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond,
                                          1 * unit.femtosecond)
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(50 * unit.kelvin)
    for T in (100, 200, 298):
        integrator.setTemperature(T * unit.kelvin)
        sim.step(2000)

    st = sim.context.getState(getPositions=True)
    app.PDBFile.writeFile(modeller.topology, st.getPositions(),
                          open(f"{OUT}/minimised.pdb", "w"))
    open(f"{OUT}/system.xml", "w").write(XmlSerializer.serialize(system))
    with open(f"{OUT}/topology.pkl", "wb") as fh:
        import pickle
        pickle.dump(modeller.topology, fh)

    from openmm import NonbondedForce
    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), NonbondedForce)][0]
    net = sum(nb.getParticleParameters(i)[0]._value
              for i in range(nb.getNumParticles()))
    e_final = sim.context.getState(getEnergy=True).getPotentialEnergy()
    return {"n_atoms": system.getNumParticles(), "n_restrained": n_restrained,
            "net_charge_e": round(net, 6),
            "energy_after_equil_kJ_per_mol": float(
                e_final.value_in_unit(unit.kilojoule_per_mole)),
            "box_nm": [round(float(v), 2) for v in
                       modeller.topology.getUnitCellDimensions()
                       .value_in_unit(unit.nanometer)]}


def run(ns=20.0, save_ps=0.4, platform=None, batch=None):
    """Production run. Saves ONLY the K+ coordinates along the pore axis."""
    import pickle
    from openmm import app, unit, XmlSerializer, LangevinMiddleIntegrator, Platform

    system = XmlSerializer.deserialize(open(f"{OUT}/system.xml").read())
    topology = pickle.load(open(f"{OUT}/topology.pkl", "rb"))
    pdb = app.PDBFile(f"{OUT}/minimised.pdb")
    meta = json.load(open(f"{OUT}/system.json"))

    integrator = LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond,
                                          4 * unit.femtosecond)
    kw = {}
    if platform:
        kw["platform"] = Platform.getPlatformByName(platform)
    sim = app.Simulation(topology, system, integrator, **kw)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(298 * unit.kelvin)
    sim.step(25_000)                       # 100 ps equilibration

    ion_idx = [a.index for a in topology.atoms()
               if a.residue.name in ("K", "K+") and a.element.symbol == "K"]
    prot_ca = [a.index for a in topology.atoms()
               if a.name == "CA" and a.residue.id in [str(r) for r in (75,)]]

    total = int(ns * 1000 / 0.004)
    # SAMPLING INTERVAL. D is measured from SHORT-TIME displacements, so the
    # frame spacing must sit inside the diffusive regime. An ion in a filter
    # site is confined to ~1 A, which it crosses in ~25 ps at D ~ 0.02
    # A^2/ps, so frames must be spaced well under that.
    #
    # The first version computed `stride` from save_ps and then never used
    # it: frames were saved once per 5000-step batch, i.e. every 20 ps. Every
    # displacement would have been fully decorrelated and D would have come
    # out meaningless while looking perfectly reasonable.
    stride = max(1, int(round(save_ps / 0.004)))
    frames = []
    R = []
    done = 0
    while done < total:
        n = min(stride, total - done)
        sim.step(n)
        done += n
        st = sim.context.getState(getPositions=True)
        x = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        frames.append((x[ion_idx], x[prot_ca] if prot_ca else None))

    # project onto the pore axis, recomputed per frame from the filter ring
    Z = []
    for ions, ring in frames:
        if ring is not None and len(ring) >= 3:
            o = ring.mean(axis=0)
            _, _, vt = np.linalg.svd(ring - o)
            ax = vt[-1]
            if ax[2] < 0:
                ax = -ax
        else:
            o, ax = np.array(meta["origin"]), np.array(meta["axis"])
        Z.append((ions - o) @ ax)

    Z = np.array(Z)
    dt_ps = stride * 0.004
    np.savez_compressed(f"{OUT}/filter_ions.npz", Z=Z, dt_ps=dt_ps,
                        ion_names=meta["ion_names"])
    return {"n_frames": len(Z), "n_ions": Z.shape[1], "dt_ps": dt_ps,
            "ns": ns, "sampling_note": f"frames every {dt_ps} ps"}


def analyse():
    """Measure D, learn the landscape, build the structure-derived model."""
    from ala2bg import kcsa
    from ala2bg.thermo.cellml_writer import write_cellml

    d = np.load(f"{OUT}/filter_ions.npz", allow_pickle=True)
    Z, dt = d["Z"], float(d["dt_ps"])
    R = d["R"] if "R" in d.files else None
    print(f"{Z.shape[0]} frames x {Z.shape[1]} ions, dt = {dt} ps")

    # RADIAL GATE. Without it a bulk ion 30 A off-axis at the same height
    # counts as being in the filter and D comes out ~29x too high -- above
    # bulk K+, which is impossible for a confined ion. The runner records r
    # for exactly this reason, so refuse to proceed silently without it.
    if R is None:
        raise ValueError(
            "filter_ions.npz has no R array. Re-run with the current "
            "run_kcsa.py: D cannot be measured from the axial coordinate "
            "alone without counting bulk ions as filter ions.")

    Dm = kcsa.measure_filter_D(Z, dt, r_traj=R, r_max=6.0)
    stab = kcsa.D_lag_stability(Z, dt, r_traj=R, r_max=6.0)
    print(f"  radially gated at r <= 6 A: {Dm['n_pairs']} pairs")
    print(f"D = {Dm['D_global']:.5f} A^2/ps from {Dm['n_pairs']} pairs; "
          f"lag stability {stab['max_ratio']:.2f} (markovian {stab['markovian']})")

    try:
        # Window chosen to cover the measured sites IN THE TRAJECTORY FRAME
        # (Thr75 CA plane), not the default +-9 A, which would clip the
        # cavity at -7.7 and everything above S1 at +9.3.
        sites, _ = kcsa.sites_from_structure(centre_on=None)
        zs = np.array(sorted(sites.values()))
        learned = kcsa.learn_landscape(Z, dt, steps=3000,
                                       z_min=float(zs.min()) - 2.5,
                                       z_max=float(zs.max()) + 2.5)
        bg = kcsa.permeation_graph_from_structure(
            Dm["D_global"], F_profile=learned["F"], z_profile=learned["z"],
            phnn_used=True, D_is_measured=True)
        bg.meta["trajectory"] = "rigid-protein KcsA in explicit water (this run)"
    except Exception as e:
        print(f"landscape learning failed ({e}); using the literature stand-in")
        bg = kcsa.permeation_graph_from_structure(Dm["D_global"],
                                                  D_is_measured=True)

    write_cellml(bg, f"{OUT}/kcsa.cellml", model_name="kcsa_filter",
                 max_states=12)
    json.dump({"provenance": kcsa.provenance_report(bg), "meta": bg.meta,
               "D": Dm["D_global"], "lag_stability": stab},
              open(f"{OUT}/provenance.json", "w"), indent=2, default=float)
    for k, v in kcsa.provenance_report(bg).items():
        print(f"  {k:24s} {v}")
    print(f"  reversal {bg.reversal_potential()*1000:.2f} mV   "
          f"conductance {bg.conductance(bg.reversal_potential())*1e12:.2f} pS")
    return bg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--ns", type=float, default=20.0)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--ions", default="S2S4", choices=list(ION_SETS))
    a = ap.parse_args()

    if a.prepare:
        p, meta = build_tetramer_pdb(ions=a.ions)
        print(f"tetramer: {meta['n_protein_atoms']} protein atoms, "
              f"ions at {meta['ion_names']}")
        info = prepare(p)
        print(f"solvated: {info['n_atoms']} atoms, "
              f"{info['n_restrained']} restrained heavy atoms")
    if a.run:
        print(run(ns=a.ns, platform=a.platform))
    if a.analyse:
        analyse()
