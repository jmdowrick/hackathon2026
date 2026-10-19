"""BondGraph -> CellML 2.0, validated.

Hand-built XML rather than ElementTree, because the namespace handling is the
part that goes wrong: CellML requires the MathML block to have the MathML
namespace as its DEFAULT namespace, and an ElementTree document that already
has a default namespace will emit <math:math>, which libcellml rejects with
"Math root node is of invalid type".

THREE BUGS THIS FILE EXISTS TO NOT HAVE AGAIN -- all three were present, all
three made the output unloadable or dimensionally wrong, and none was visible
without running the official validator:

  1. <math:math> prefix instead of a default MathML namespace. Every MathML
     child then failed to resolve and the model would not open at all.
  2. Redefining 'mole' and 'second', which are protected standard unit names
     in CellML 2.0.
  3. kappa declared as mol^-1 s^-1. From v = kappa (K_a q_a - K_b q_b) with K
     in mol^-1 and q in mol, the bracket is DIMENSIONLESS, so kappa carries
     the units of v: mol s^-1. Hunter et al. write the same form with
     exponentials, where the bracket is manifestly dimensionless.

test_cellml_validates_with_libcellml pins all three against the real
validator, so this cannot silently regress.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from .smoluchowski import rt

CELLML_NS = "http://www.cellml.org/cellml/2.0#"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# Faraday constant expressed so that z * F * V lands in kJ/mol directly.
FARADAY_kJ_per_mol_per_V = 96.48533212

# Units we must define. 'mole', 'second', 'joule', 'volt', 'ampere',
# 'coulomb' and 'dimensionless' are CellML 2.0 standard units and MUST NOT be
# redefined.
_UNIT_DEFS = {
    "kJ_per_mol": [("joule", {"multiplier": "1000"}), ("mole", {"exponent": "-1"})],
    "mol_per_s": [("mole", {}), ("second", {"exponent": "-1"})],
    "per_mole": [("mole", {"exponent": "-1"})],
    "kJ_per_mol_per_V": [("joule", {"multiplier": "1000"}),
                         ("mole", {"exponent": "-1"}),
                         ("volt", {"exponent": "-1"})],
    "C_per_mol": [("coulomb", {}), ("mole", {"exponent": "-1"})],
    "per_second": [("second", {"exponent": "-1"})],
}


def _units_block(names):
    out = []
    for n in names:
        rows = "".join(
            f'    <unit units="{u}"'
            + "".join(f' {k}="{v}"' for k, v in attrs.items()) + " />\n"
            for u, attrs in _UNIT_DEFS[n])
        out.append(f'  <units name="{n}">\n{rows}  </units>\n')
    return "".join(out)


def _var(name, units, initial=None):
    init = f' initial_value="{initial!r}"' if initial is not None else ""
    return f'    <variable name="{escape(name)}" units="{units}"{init} />\n'


def _ci(n):
    return f"<ci>{escape(str(n))}</ci>"


def _apply(op, *kids):
    return f"<apply><{op}/>{''.join(kids)}</apply>"


def _eq(lhs, rhs):
    return _apply("eq", lhs, rhs)


def _ode(var, t="t"):
    return f"<apply><diff/><bvar>{_ci(t)}</bvar>{_ci(var)}</apply>"


def _safe(labels, n):
    """Component-safe identifiers: CellML names must be valid XML names."""
    out, seen = [], set()
    for i in range(n):
        raw = str(labels[i]) if labels and i < len(labels) else f"s{i}"
        s = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw)
        if not s or not (s[0].isalpha() or s[0] == "_"):
            s = "s" + s
        while s in seen:
            s += "_"
        seen.add(s)
        out.append(s)
    return out


def write_cellml(bg, path, model_name="bond_graph_model", max_states=None,
                 validate=True):
    """Emit CellML 2.0 for a bond graph.

    If the graph carries per-reaction charges (an ElectroBondGraph), the
    electrical port is emitted too: membrane potential as a variable, the
    exponential factor on the forward term, and the transmembrane current.
    """
    n = bg.n_states
    if max_states is not None and n > max_states:
        raise ValueError(
            f"{n} states exceeds max_states={max_states}; "
            "pass max_states=None to emit the fine model deliberately")

    z = getattr(bg, "z", None)
    A = getattr(bg, "affinity", None)
    electro = z is not None
    if electro and A is None:
        A = np.zeros(len(z))

    # Explicit species (bath concentrations). Emitted as CellML variables with
    # their own potentials, so [K+] is something a user can CHANGE in OpenCOR
    # and something a cell model can drive -- rather than a constant folded
    # into the affinity at build time, which is what this previously did.
    species = getattr(bg, "species", None) or {}
    transits = getattr(bg, "transits", None) or []
    sp_names = _safe(list(species.keys()), len(species))
    sp_map = dict(zip(species.keys(), sp_names))

    lab = _safe(bg.labels, n)
    RT = rt(getattr(bg, "temperature", 298.0))

    # UNITS FOLLOW THE MODE. A single-channel model has a dimensionless
    # occupancy, fluxes in s^-1, and one ELEMENTARY charge per hop. A molar
    # model (Hunter's convention) has amounts in mol, fluxes in mol/s, and the
    # Faraday constant.
    #
    # Emitting the molar charge constant for a single-channel model overstates
    # the current by Avogadro's number -- the same bug that produced 1e21 pS
    # in electro.py, reintroduced here because the writer hardcoded Faraday
    # and ignored bg.mode. A current of ~1e13 A on a channel is the symptom.
    molar = getattr(bg, "mode", "molar") == "molar"
    u_q = "mole" if molar else "dimensionless"
    u_K = "per_mole" if molar else "dimensionless"
    u_v = "mol_per_s" if molar else "per_second"

    used = ["kJ_per_mol", "per_mole", "mol_per_s", "per_second"]
    if electro:
        used += ["kJ_per_mol_per_V", "C_per_mol"]

    # --- variables ---------------------------------------------------------
    v = [_var("t", "second"), _var("RT", "kJ_per_mol", RT)]
    if electro:
        v += [_var("V", "volt", 0.0),
              _var("Far", "kJ_per_mol_per_V", FARADAY_kJ_per_mol_per_V)]
        if molar:
            v += [_var("q_e", "C_per_mol", 96485.33212)]
        else:
            v += [_var("q_e", "coulomb", 1.602176634e-19)]
        v += [_var("I", "ampere")]

    x0 = 1.0 / n
    for i in range(n):
        v += [_var(f"q_{lab[i]}", u_q, x0),
              _var(f"mu_{lab[i]}", "kJ_per_mol"),
              _var(f"K_{lab[i]}", u_K, float(bg.K[i]))]
    for raw, nm in sp_map.items():
        sp = species[raw]
        v += [_var(f"q_{nm}", u_q, float(sp.q)),
              _var(f"K_{nm}", u_K, float(sp.K)),
              _var(f"mu_{nm}", "kJ_per_mol")]

    for r in range(bg.n_reactions):
        # kappa carries the units of v: the (K q) bracket is dimensionless
        v += [_var(f"v_{r}", u_v),
              _var(f"kappa_{r}", u_v, float(bg.kappa[r]))]
        if electro:
            v += [_var(f"z_{r}", "dimensionless", float(z[r]))]
            if abs(float(A[r])) > 0:
                # electro.py stores the chemostat affinity in J/mol (it is
                # divided by RT*1000 there). The CellML declares kJ/mol to
                # match RT, so it MUST be converted. Declaring J/mol numbers
                # as kJ/mol passes libcellml -- the units are internally
                # consistent -- and silently scales the driving force by 1000.
                v += [_var(f"A_{r}", "kJ_per_mol", float(A[r]) / 1000.0)]

    # --- equations ---------------------------------------------------------
    eqs = []
    for i in range(n):
        eqs.append(_eq(_ci(f"mu_{lab[i]}"),
                       _apply("times", _ci("RT"),
                              _apply("ln", _apply("times", _ci(f"K_{lab[i]}"),
                                                  _ci(f"q_{lab[i]}"))))))

    for raw, nm in sp_map.items():
        eqs.append(_eq(_ci(f"mu_{nm}"),
                       _apply("times", _ci("RT"),
                              _apply("ln", _apply("times", _ci(f"K_{nm}"),
                                                  _ci(f"q_{nm}"))))))

    transit_by_r = {t.reaction: t for t in transits}

    for r, (a, b) in enumerate(bg.edges):
        fwd = _apply("times", _ci(f"K_{lab[a]}"), _ci(f"q_{lab[a]}"))
        if electro:
            # exp((z_r Far V + A_r) / RT)
            drive = _apply("times", _ci(f"z_{r}"), _ci("Far"), _ci("V"))
            if abs(float(A[r])) > 0:
                drive = _apply("plus", drive, _ci(f"A_{r}"))
            t = transit_by_r.get(r)
            if t is not None:
                # A_r = mu_source - mu_sink, as variables rather than a constant
                drive = _apply("plus", drive,
                               _apply("minus", _ci(f"mu_{sp_map[t.source]}"),
                                      _ci(f"mu_{sp_map[t.sink]}")))
            fwd = _apply("times", fwd,
                         _apply("exp", _apply("divide", drive, _ci("RT"))))
        rev = _apply("times", _ci(f"K_{lab[b]}"), _ci(f"q_{lab[b]}"))
        eqs.append(_eq(_ci(f"v_{r}"),
                       _apply("times", _ci(f"kappa_{r}"),
                              _apply("minus", fwd, rev))))

    terms = {i: [] for i in range(n)}
    for r, (a, b) in enumerate(bg.edges):
        terms[a].append((-1.0, r))
        terms[b].append((1.0, r))
    for i in range(n):
        tl = terms[i]
        if not tl:
            rhs = ('<cn cellml:units="%s" xmlns:cellml="%s">0</cn>'
                   % (u_v, CELLML_NS))
        else:
            parts = [_ci(f"v_{r}") if s > 0 else _apply("minus", _ci(f"v_{r}"))
                     for s, r in tl]
            rhs = parts[0] if len(parts) == 1 else _apply("plus", *parts)
        eqs.append(_eq(_ode(f"q_{lab[i]}"), rhs))

    if electro:
        # I = F sum_r z_r v_r
        prods = [_apply("times", _ci(f"z_{r}"), _ci(f"v_{r}"))
                 for r in range(bg.n_reactions)]
        tot = prods[0] if len(prods) == 1 else _apply("plus", *prods)
        eqs.append(_eq(_ci("I"), _apply("times", _ci("q_e"), tot)))

    math_block = (f'    <math xmlns="{MATHML_NS}">\n      '
                  + "\n      ".join(eqs) + "\n    </math>\n")

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model xmlns="{CELLML_NS}" name="{escape(model_name)}">\n'
        + _units_block(used)
        + '  <component name="bond_graph">\n'
        + "".join(v) + math_block
        + "  </component>\n</model>\n"
    )

    with open(path, "w") as fh:
        fh.write(xml)

    if validate:
        issues = validate_cellml(path)
        if issues:
            raise ValueError("emitted CellML does not validate:\n  "
                             + "\n  ".join(issues[:8]))
    return path


def validate_cellml(path):
    """Validate with libcellml. Returns a list of issue descriptions.

    Returns ["libcellml not installed"] rather than silently passing when the
    validator is unavailable -- a validation step that quietly does nothing is
    worse than none.
    """
    try:
        import libcellml
    except ImportError:                                  # pragma: no cover
        return ["libcellml not installed"]

    text = open(path).read()
    p = libcellml.Parser()
    m = p.parseModel(text)
    out = [p.issue(i).description() for i in range(p.issueCount())]
    if m is not None:
        v = libcellml.Validator()
        v.validateModel(m)
        out += [v.issue(i).description() for i in range(v.issueCount())]
    return out
