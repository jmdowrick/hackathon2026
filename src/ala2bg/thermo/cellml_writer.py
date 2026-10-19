"""BondGraph -> CellML 2.0.

Hand-rolled XML, no third-party runtime dependency. BondGraphTools is
effectively unmaintained and its sympy pinning is a known hazard; a writer we
control cannot break on someone else's release.

The emitted model is the master equation in Ce/Re form:

    mu_i = RT * ln(K_i * q_i)
    v_r  = kappa_r * ( K_a q_a - K_b q_b )
    dq_i/dt = sum_r N_ir v_r

test_cellml_roundtrip.py integrates this and compares against the generator,
which is the real proof of correctness; OpenCOR is presentation polish.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .smoluchowski import rt

CELLML_NS = "http://www.cellml.org/cellml/2.0#"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"


def _ci(name):
    e = ET.Element(f"{{{MATHML_NS}}}ci"); e.text = f" {name} "
    return e


def _cn(value, units="dimensionless"):
    e = ET.Element(f"{{{MATHML_NS}}}cn")
    e.set(f"{{{CELLML_NS}}}units", units)
    e.text = f" {value!r} "
    return e


def _apply(op, *kids):
    e = ET.Element(f"{{{MATHML_NS}}}apply")
    e.append(ET.Element(f"{{{MATHML_NS}}}{op}"))
    for k in kids:
        e.append(k)
    return e


def write_cellml(bg, path, model_name="bond_graph_model", max_states=None):
    """Emit CellML 2.0 for the bond graph.

    max_states guards against emitting a 3600-state model by accident; pass
    None deliberately for the fine model.
    """
    n = bg.n_states
    if max_states is not None and n > max_states:
        raise ValueError(
            f"{n} states exceeds max_states={max_states}; "
            "pass max_states=None to emit the fine model deliberately"
        )

    ET.register_namespace("", CELLML_NS)
    ET.register_namespace("math", MATHML_NS)

    model = ET.Element(f"{{{CELLML_NS}}}model", {"name": model_name})

    # --- units -------------------------------------------------------------
    for uname, parts in [
        ("mole", [{"units": "mole"}]),
        ("second", [{"units": "second"}]),
        ("kJ_per_mol", [{"units": "joule", "multiplier": "1000"},
                        {"units": "mole", "exponent": "-1"}]),
        ("mol_per_s", [{"units": "mole"}, {"units": "second", "exponent": "-1"}]),
        ("per_mol_per_s", [{"units": "mole", "exponent": "-1"},
                           {"units": "second", "exponent": "-1"}]),
        ("per_mole", [{"units": "mole", "exponent": "-1"}]),
    ]:
        u = ET.SubElement(model, f"{{{CELLML_NS}}}units", {"name": uname})
        for p in parts:
            ET.SubElement(u, f"{{{CELLML_NS}}}unit", p)

    comp = ET.SubElement(model, f"{{{CELLML_NS}}}component", {"name": "bond_graph"})

    def var(name, units, **kw):
        a = {"name": name, "units": units}
        a.update({k: str(v) for k, v in kw.items()})
        ET.SubElement(comp, f"{{{CELLML_NS}}}variable", a)

    var("t", "second")
    var("RT", "kJ_per_mol", initial_value=repr(rt(bg.temperature)))

    labels = bg.labels or [f"s{i}" for i in range(n)]
    labels = [str(l).replace(" ", "_") for l in labels]

    x0 = np.full(n, 1.0 / n)
    for i in range(n):
        var(f"q_{labels[i]}", "mole", initial_value=repr(float(x0[i])))
        var(f"mu_{labels[i]}", "kJ_per_mol")
        var(f"K_{labels[i]}", "per_mole", initial_value=repr(float(bg.K[i])))
    for r in range(bg.n_reactions):
        var(f"v_{r}", "mol_per_s")
        var(f"kappa_{r}", "per_mol_per_s", initial_value=repr(float(bg.kappa[r])))

    math_el = ET.SubElement(comp, f"{{{MATHML_NS}}}math")

    # mu_i = RT * ln(K_i * q_i)
    for i in range(n):
        math_el.append(_apply(
            "eq", _ci(f"mu_{labels[i]}"),
            _apply("times", _ci("RT"),
                   _apply("ln", _apply("times", _ci(f"K_{labels[i]}"),
                                       _ci(f"q_{labels[i]}")))),
        ))

    # v_r = kappa_r * (K_a q_a - K_b q_b)
    for r, (a, b) in enumerate(bg.edges):
        math_el.append(_apply(
            "eq", _ci(f"v_{r}"),
            _apply("times", _ci(f"kappa_{r}"),
                   _apply("minus",
                          _apply("times", _ci(f"K_{labels[a]}"), _ci(f"q_{labels[a]}")),
                          _apply("times", _ci(f"K_{labels[b]}"), _ci(f"q_{labels[b]}")))),
        ))

    # dq_i/dt = sum_r N_ir v_r
    terms = {i: [] for i in range(n)}
    for r, (a, b) in enumerate(bg.edges):
        terms[a].append((-1.0, r))
        terms[b].append((+1.0, r))
    for i in range(n):
        diff = ET.Element(f"{{{MATHML_NS}}}apply")
        diff.append(ET.Element(f"{{{MATHML_NS}}}diff"))
        bvar = ET.SubElement(diff, f"{{{MATHML_NS}}}bvar"); bvar.append(_ci("t"))
        diff.append(_ci(f"q_{labels[i]}"))

        tl = terms[i]
        if not tl:
            rhs = _cn(0.0, "mol_per_s")
        else:
            parts = [_ci(f"v_{r}") if s > 0
                     else _apply("minus", _ci(f"v_{r}")) for s, r in tl]
            rhs = parts[0] if len(parts) == 1 else _apply("plus", *parts)
        math_el.append(_apply("eq", diff, rhs))

    tree = ET.ElementTree(model)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
