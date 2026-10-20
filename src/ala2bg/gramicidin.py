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
