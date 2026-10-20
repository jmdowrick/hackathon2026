"""Molecular trajectories to runnable physiological models.

Learns a free-energy landscape and a dissipation field from MD with
thermodynamic structure built into the architecture, then discretises them
into a port-Hamiltonian bond graph exported as CellML 2.0.

Validated on alanine dipeptide (where ground truth is known) and applied to
the KcsA potassium channel, with geometry measured from the PDB and an
electrical port giving membrane voltage and ionic current.

The package name is historical: the project began on alanine dipeptide.
"""
__version__ = "0.1.0"
