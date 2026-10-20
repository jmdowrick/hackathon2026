"""Gramicidin A: the target where the one-ion bond graph is the right model.

WHY GA. KcsA conducts by multi-ion knock-on, so a single-ion bond graph is
structurally the wrong model there (HANDOVER section 4). GA is a single-file
pore, predominantly singly occupied at physiological concentration, so the
same bond graph is the CORRECT structure rather than an approximation. It is
~25k atoms in a bilayer and still electrogenic.

What this module does NOT know, deliberately: the binding-site positions.
1JNO is an NMR structure in SDS micelles with no resolved ions, so unlike
1K4C there is nothing to read the sites off. They are for the SRV/MSM to
discover; the published cation sites near +-9.6 A are a cross-check, not an
input.

FRAME. Origin at the dimer junction (midpoint of the two N-terminal formyl-
valines); z along the helix axis. GA is a symmetric dimer, so which end is
"outside" is a LABELLING choice, not physics: +z is taken towards chain A's
C-terminus and recorded, so a later sign slip shows up as a frame mismatch
rather than as a quietly reversed current.

SCOPE OF 1JNO. Micelle, not bilayer; NMR minimised average, not dynamics. It
fixes the pore frame and a starting radius profile. It does not fix the
landscape, and an SDS micelle is not a membrane.
"""

from __future__ import annotations

import numpy as np

from . import pdb

GA_PDB = "1JNO"
# Published cation-binding sites, for cross-checking discovered states only.
# Literature values cluster near +-9.6 A from the dimer centre; treat as
# approximate (they vary with method and ion by ~1 A).
PUBLISHED_SITE_Z_A = (-9.6, 9.6)


def load_gramicidin(pdb_id=GA_PDB, cache_dir="data", n_slabs=80,
                    z_range=(-20.0, 20.0), probe_A=1.4):
    """Structure -> pore frame and radius profile, measured from the PDB."""
    path = pdb.fetch(pdb_id, cache_dir)
    s, _ = pdb.read_pdb(path, keep_chains=["A", "B"])
    if len(set(s.chain)) != 2:
        raise ValueError(f"expected the two GA monomers, got chains "
                         f"{sorted(set(s.chain))}")

    # Axis: the long direction of the backbone. The beta-helix dimer is ~25 A
    # long and ~4 A across, so the first singular vector is unambiguous; the
    # ratio is reported so a structure where it is not can be noticed.
    bb = s.select(name=["N", "CA", "C"])
    X = s.xyz[bb]
    _, sv, vt = np.linalg.svd(X - X.mean(axis=0))
    axis = vt[0] / np.linalg.norm(vt[0])

    nterm = [s.xyz[s.select(chain=c, resname="FVA")].mean(axis=0) for c in "AB"]
    cterm = [s.xyz[s.select(chain=c, resname="ETA")].mean(axis=0) for c in "AB"]
    # The junction fixes WHERE along the axis the origin sits, not where the
    # axis is: the formyl-valines are off-axis, and an origin taken at their
    # midpoint displaces the line every radius is measured from -- the lumen
    # then reads as clipped to zero in places. Project onto the axis line
    # through the backbone centroid.
    c = X.mean(axis=0)
    junction = 0.5 * (nterm[0] + nterm[1])
    origin = c + np.dot(junction - c, axis) * axis
    if np.dot(cterm[0] - origin, axis) < 0:
        axis = -axis                        # +z towards chain A's C-terminus

    wall = pdb.non_solvent(s)
    edges = np.linspace(z_range[0], z_range[1], n_slabs + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    rad = pdb.radius_profile(s, origin, axis, edges, probe_A, wall_mask=wall)

    def z_of(p):
        return float(np.dot(p - origin, axis))

    return {
        "pdb_id": pdb_id.upper(), "path": path,
        "origin": origin, "axis": axis,
        "axis_anisotropy": float(sv[0] / sv[1]),
        "nterm_z_A": [z_of(p) for p in nterm],
        "cterm_z_A": [z_of(p) for p in cterm],
        "radius_z_A": centres, "radius_A": rad,
        "n_hetatm_in_wall": int((wall & s.is_hetatm).sum()),
        "frame": "origin at FVA1 midpoint; +z towards chain A C-terminus",
        "note": ("Radius from ALL non-solvent heavy atoms: GA's D-residues "
                 "and caps are HETATM. Sites are not taken from this file."),
    }


def channel_from_structure(g, r_gate_A=4.0, margin_A=2.0):
    """A features.Channel spanning the pore plus a margin at each mouth.

    r_gate_A: the radial gate. GA's lumen is ~2 A in radius; 4 A admits an ion
    in the pore with its first water shell partly stripped, and excludes ions
    sitting on the lipid headgroups beside it.
    """
    from .features import Channel
    zc = sorted(g["cterm_z_A"])
    return Channel(zc[0] - margin_A, zc[1] + margin_A, r_gate_A)


# ---------------------------------------------------------------------------
# Force field: Amber ff14SB + lipid17, with two cap templates
# ---------------------------------------------------------------------------
#
# WHY AMBER, AND WHY THE D-RESIDUES NEED NO PARAMETERS. Every ff14SB proper
# torsion has phase 0 or pi and there is no CMAP, so every term in the energy
# is EVEN under mirror inversion. A D-residue on the L template is therefore
# the exact mirror image of the L-residue -- not an approximation.
# test_ga_energy_is_mirror_invariant checks it on the real dimer. CHARMM36's
# CMAP is fitted to L-residues and would not have this property.
#
# THE CAPS. Every bonded term already exists in ff14SB: H5 is Amber's formyl
# hydrogen (C-H5, H5-C-N, H5-C-O), and ethanolamine is serine-like CT/OH.
# Only the CHARGES are new, and they are by ANALOGY, not a RESP fit:
#   formyl       = ACE with the methyl group's net charge (-0.0293) on HN
#   ethanolamine = NME amide (N, H) + serine CB/OG/HG, with CA set so the
#                  residue is neutral
# The formyls sit at the dimer junction (the central barrier) and the
# ethanolamines at the mouths (the entrance sites), so these charges are a
# declared, testable choice, not a detail.

GA_CAP_TEMPLATES = """<ForceField>
 <Residues>
  <Residue name="FVA">
   <Atom name="CN"   type="protein-C"  charge="0.5972"/>
   <Atom name="O1"   type="protein-O"  charge="-0.5679"/>
   <Atom name="HN"   type="protein-H5" charge="-0.0293"/>
   <Atom name="N"    type="protein-N"  charge="-0.4157"/>
   <Atom name="H"    type="protein-H"  charge="0.2719"/>
   <Atom name="CA"   type="protein-CX" charge="-0.0875"/>
   <Atom name="HA"   type="protein-H1" charge="0.0969"/>
   <Atom name="CB"   type="protein-3C" charge="0.2985"/>
   <Atom name="HB"   type="protein-HC" charge="-0.0297"/>
   <Atom name="CG1"  type="protein-CT" charge="-0.3192"/>
   <Atom name="HG11" type="protein-HC" charge="0.0791"/>
   <Atom name="HG12" type="protein-HC" charge="0.0791"/>
   <Atom name="HG13" type="protein-HC" charge="0.0791"/>
   <Atom name="CG2"  type="protein-CT" charge="-0.3192"/>
   <Atom name="HG21" type="protein-HC" charge="0.0791"/>
   <Atom name="HG22" type="protein-HC" charge="0.0791"/>
   <Atom name="HG23" type="protein-HC" charge="0.0791"/>
   <Atom name="C"    type="protein-C"  charge="0.5973"/>
   <Atom name="O"    type="protein-O"  charge="-0.5679"/>
   <Bond atomName1="CN" atomName2="O1"/>
   <Bond atomName1="CN" atomName2="HN"/>
   <Bond atomName1="CN" atomName2="N"/>
   <Bond atomName1="N" atomName2="H"/>
   <Bond atomName1="N" atomName2="CA"/>
   <Bond atomName1="CA" atomName2="HA"/>
   <Bond atomName1="CA" atomName2="CB"/>
   <Bond atomName1="CA" atomName2="C"/>
   <Bond atomName1="CB" atomName2="HB"/>
   <Bond atomName1="CB" atomName2="CG1"/>
   <Bond atomName1="CB" atomName2="CG2"/>
   <Bond atomName1="CG1" atomName2="HG11"/>
   <Bond atomName1="CG1" atomName2="HG12"/>
   <Bond atomName1="CG1" atomName2="HG13"/>
   <Bond atomName1="CG2" atomName2="HG21"/>
   <Bond atomName1="CG2" atomName2="HG22"/>
   <Bond atomName1="CG2" atomName2="HG23"/>
   <Bond atomName1="C" atomName2="O"/>
   <ExternalBond atomName="C"/>
  </Residue>
  <Residue name="ETA">
   <Atom name="N"   type="protein-N"  charge="-0.4157"/>
   <Atom name="H"   type="protein-H"  charge="0.2719"/>
   <Atom name="CA"  type="protein-CT" charge="-0.1064"/>
   <Atom name="HA1" type="protein-H1" charge="0.0976"/>
   <Atom name="HA2" type="protein-H1" charge="0.0976"/>
   <Atom name="C"   type="protein-CT" charge="0.2117"/>
   <Atom name="HB1" type="protein-H1" charge="0.0352"/>
   <Atom name="HB2" type="protein-H1" charge="0.0352"/>
   <Atom name="O"   type="protein-OH" charge="-0.6546"/>
   <Atom name="HO"  type="protein-HO" charge="0.4275"/>
   <Bond atomName1="N" atomName2="H"/>
   <Bond atomName1="N" atomName2="CA"/>
   <Bond atomName1="CA" atomName2="HA1"/>
   <Bond atomName1="CA" atomName2="HA2"/>
   <Bond atomName1="CA" atomName2="C"/>
   <Bond atomName1="C" atomName2="HB1"/>
   <Bond atomName1="C" atomName2="HB2"/>
   <Bond atomName1="C" atomName2="O"/>
   <Bond atomName1="O" atomName2="HO"/>
   <ExternalBond atomName="N"/>
  </Residue>
 </Residues>
</ForceField>
"""

# D-residues use the L templates: exact, by the mirror argument above.
D_TO_L = {"DLE": "LEU", "DVA": "VAL"}


def forcefield():
    """amber14-all (ff14SB + lipid17) + TIP3P + the GA cap templates."""
    import io
    from openmm import app
    return app.ForceField("amber14-all.xml", "amber14/tip3p.xml",
                          io.StringIO(GA_CAP_TEMPLATES))


def openmm_model(g=None):
    """GA as an OpenMM (topology, positions), rotated into the pore frame.

    The membrane builder takes the membrane normal as z and centres the
    bilayer at z = 0, so the structure is placed with the channel axis on z
    and the dimer junction at the origin -- the same frame as
    load_gramicidin(), which is what lets features and delta be read off a
    trajectory without a frame conversion (bug 3 was a 5.89 A one).
    """
    import io
    from openmm import app, unit

    g = g or load_gramicidin()
    lines = []
    for line in open(g["path"]):
        if line.startswith(("ATOM", "HETATM")):
            if line[21] not in "AB":
                continue
            rn = line[17:20]
            if rn in D_TO_L:
                line = line[:17] + D_TO_L[rn] + line[20:]
        elif line.startswith(("CONECT", "LINK", "MODEL", "ENDMDL", "TER")):
            continue
        lines.append(line)
    pdbf = app.PDBFile(io.StringIO("".join(lines)))
    top = pdbf.topology

    # Rotation taking the pore axis to +z.
    a = np.asarray(g["axis"], float)
    ez = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, ez), float(np.dot(a, ez))
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)
    X = pdbf.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    X = (X - g["origin"]) @ R.T

    # Bonds the templates need but PDBFile does not make: within the caps,
    # and the peptide links into and out of them. By distance, so nothing is
    # taken on trust from CONECT records that were stripped above.
    atoms = list(top.atoms())
    have = {frozenset((b[0].index, b[1].index)) for b in top.bonds()}
    caps = [i for i, at in enumerate(atoms) if at.residue.name in ("FVA", "ETA")]
    for i in caps:
        for j, at in enumerate(atoms):
            if j == i or frozenset((i, j)) in have:
                continue
            ei, ej = atoms[i].element.symbol, at.element.symbol
            cut = 1.25 if "H" in (ei, ej) else 1.75
            same_chain = atoms[i].residue.chain is at.residue.chain
            if same_chain and np.linalg.norm(X[i] - X[j]) < cut:
                top.addBond(atoms[i], at)
                have.add(frozenset((i, j)))
    pos = unit.Quantity([tuple(x) for x in X / 10.0], unit.nanometer)
    return top, pos


def unexplained_missing_torsions(top, system, ff,
                                 ff_file="amber14/protein.ff14SB.xml"):
    """Proper torsions absent from the system that the force field does not
    explicitly set to zero.

    OpenMM drops a proper torsion SILENTLY when no parameter matches -- and
    also, legitimately, when the matching parameter has zero barrier (ff14SB
    sets e.g. X-CT-N-X to k = 0, as Amber's parm10 does). Only the first is a
    defect, and createSystem cannot tell them apart. This can: every torsion
    in the bond graph must be either in the system or matched by an explicit
    k = 0 entry. Returns the ones that are neither, as atom-name strings.
    """
    import os
    import xml.etree.ElementTree as ET
    import openmm as mm
    from openmm import app

    root = ET.parse(os.path.join(os.path.dirname(app.__file__), "data",
                                 ff_file)).getroot()
    zero = []
    for p in root.find("PeriodicTorsionForce"):
        if p.tag != "Proper":
            continue
        ks = [float(v) for k, v in p.attrib.items() if k.startswith("k")]
        if ks and all(k == 0.0 for k in ks):
            zero.append(tuple(p.get(f"type{i}") or "" for i in (1, 2, 3, 4)))

    atype = {}
    for res, tpl in zip(top.residues(), ff.getMatchingTemplates(top)):
        byname = {a.name: a.type for a in tpl.atoms}
        for at in res.atoms():
            atype[at.index] = byname[at.name]

    def matches_zero(q):
        t = tuple(atype[i] for i in q)
        for z in zero:
            for tt in (t, t[::-1]):
                if all(zz == "" or zz == x for zz, x in zip(z, tt)):
                    return True
        return False

    tf = [f for f in system.getForces()
          if isinstance(f, mm.PeriodicTorsionForce)][0]
    have = set()
    for n in range(tf.getNumTorsions()):
        q = tuple(tf.getTorsionParameters(n)[:4])
        have.add(min(q, q[::-1]))

    nbr = {a.index: set() for a in top.atoms()}
    for b in top.bonds():
        nbr[b[0].index].add(b[1].index)
        nbr[b[1].index].add(b[0].index)
    atoms = list(top.atoms())
    bad, seen = [], set()
    for j in nbr:
        for k in nbr[j]:
            for i in nbr[j] - {k}:
                for l in nbr[k] - {j, i}:
                    q = min((i, j, k, l), (l, k, j, i))
                    if q in seen:
                        continue
                    seen.add(q)
                    if q not in have and not matches_zero(q):
                        bad.append("-".join(f"{atoms[x].residue.name}:"
                                            f"{atoms[x].name}" for x in q))
    return bad
