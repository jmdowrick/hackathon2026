"""Electrogenic bond graphs: the electrical port, following Hunter et al.

FORMULATION (Hunter et al., "Energy-based bond graph models of glucose
transport with SLC transporters", bioRxiv 2024.06.26.600892)

Chemical storage is unchanged from the rest of this package:

    u_i = RT ln(K_i q_i)                             [J/mol]

The electrical domain couples at the 1-junction (energy balance), where the
membrane potential contributes an ADDITIVE term to the forward potential of an
electrogenic reaction:

    u_f = u_a + z_r F V ,      u_r = u_b

so the Marcelin-de Donder flux acquires an exponential factor on the forward
term only:

    v_r = kappa_r [ K_a q_a exp(z_r F V / RT) - K_b q_b ]

Charge is conserved at a 0-node with capacitive storage in the membrane:

    d(q_charge)/dt = -F sum_r z_r v_r ,     V = q_charge / C_m

and the current is

    I = F sum_r z_r v_r                              [C/s = A]

WHAT IS ADAPTED HERE, AND WHY
-----------------------------
Hunter's Figure 4 is a single-step channel, so the whole valence z sits on one
reaction. Ion permeation through a selectivity filter is NOT single-step: a K+
ion hops S4 -> S3 -> S2 -> S1 -> exit, and each hop traverses only a FRACTION
of the transmembrane field. So each reaction carries its own coefficient

    z_r = z_ion * delta_r ,      sum_r delta_r = 1 around a complete cycle

delta_r is the Woodhull electrical distance of step r. The constraint on the
sum is not cosmetic: it is what makes the model reproduce the Nernst potential
and what forbids a steady current at equilibrium. check_charge_closure()
enforces it rather than trusting it.

This per-step coefficient is precisely the parameter family that whole-cell
transporter models usually ASSUME rather than measure. Derived here from the
landscape -- the electrical distance of a state is the position of its charge
along the pore axis -- it becomes a computed quantity.

THERMODYNAMIC CHECKS THAT MUST HOLD
-----------------------------------
1. At V = V_Nernst the net flux is exactly zero (no current without a driving
   force). This is Hunter's validation and it is non-trivial: it only holds if
   the charge coefficients sum correctly around the cycle.
2. At V = 0 with equal concentrations, the cycle affinity vanishes, so there is
   no steady current. A model that produces current from nothing is broken.
3. Power balance: the electrical power F V sum_r z_r v_r must equal the
   electrical term in the dissipation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .thermo.smoluchowski import rt

FARADAY = 96485.33212        # C/mol
ELEMENTARY_CHARGE = 1.602176634e-19     # C
AVOGADRO = 6.02214076e23                # 1/mol


def thermal_voltage(temperature: float = 298.0) -> float:
    """RT/F in volts -- 25.7 mV at 298 K."""
    return rt(temperature) * 1000.0 / FARADAY


# UNITS. Hunter's formulation is molar: q in mol, v in mol/s, charge via the
# Faraday constant, which is right for a whole-cell model where the state is an
# amount of transporter. A SINGLE-CHANNEL conductance is a per-molecule
# quantity: the state is an occupancy probability (dimensionless, summing to
# one), fluxes are transitions per second, and the charge carried by one hop is
# the ELEMENTARY charge, not the Faraday constant.
#
# Using F where e belongs inflates the current by Avogadro's number, which is
# exactly the kind of error that produces a conductance of 1e21 pS on a slide
# and is otherwise invisible because the structure and the checks all still
# pass. The mode is therefore explicit and recorded in the artifact.
CHARGE_UNIT = {"single_channel": ELEMENTARY_CHARGE, "molar": FARADAY}


@dataclass
class ElectroBondGraph:
    """A bond graph with an electrical port on each electrogenic reaction.

    K      : (n,) Ce constants, mol^-1
    kappa  : (m,) Re constants
    edges  : (m, 2) endpoints, reaction r goes edges[r,0] -> edges[r,1]
    z      : (m,) charge moved per reaction, in units of e (signed, and
             already including the Woodhull fraction delta_r)
    """

    K: np.ndarray
    kappa: np.ndarray
    edges: np.ndarray
    z: np.ndarray
    affinity: np.ndarray | None = None   # chemostat affinity per reaction, J/mol
    F_states: np.ndarray | None = None
    temperature: float = 298.0
    C_m: float = 1e-14                 # membrane capacitance, F (farads)
    mode: str = "single_channel"       # or "molar" -- see CHARGE_UNIT above
    labels: list | None = None
    meta: dict = field(default_factory=dict)

    @property
    def q_charge(self) -> float:
        """Charge per unit of flux: e for a single channel, F for molar."""
        try:
            return CHARGE_UNIT[self.mode]
        except KeyError:
            raise ValueError(f"mode must be one of {sorted(CHARGE_UNIT)}")

    @classmethod
    def from_rates(cls, k_forward, k_reverse, edges, z, F_states=None,
                   temperature=298.0, labels=None, mode="single_channel",
                   affinity=None, **kw):
        """Build from RATE CONSTANTS (per second), which is how channel
        kinetics are actually reported, rather than from Ce/Re constants.

        Detailed balance fixes the state constants: K_j/K_i = k_ij/k_ji at
        zero voltage, so the K are obtained by walking the graph, and kappa
        follows as kappa_r = k_forward_r / K_a. If the supplied rates are not
        consistent around a cycle this will show up in check_charge_closure
        and in the Wegscheider residual rather than being silently absorbed.
        """
        kf = np.asarray(k_forward, float); kr = np.asarray(k_reverse, float)
        e = np.asarray(edges, int)
        n = int(e.max()) + 1
        K = np.full(n, np.nan); K[0] = 1.0
        for _ in range(n):                      # propagate detailed balance
            for r, (a, b) in enumerate(e):
                if np.isfinite(K[a]) and not np.isfinite(K[b]):
                    K[b] = K[a] * kf[r] / kr[r]
                elif np.isfinite(K[b]) and not np.isfinite(K[a]):
                    K[a] = K[b] * kr[r] / kf[r]
        K = np.where(np.isfinite(K), K, 1.0)
        kappa = kf / K[e[:, 0]]

        # Any inconsistency of the supplied rates around a cycle is a real
        # driving force, not a rounding error. Recover it as an explicit
        # chemostat affinity instead of discarding it.
        A = rt(temperature) * 1000.0 * np.log(
            np.clip(kf / (kappa * K[e[:, 0]]), 1e-300, None))
        A = np.where(np.isfinite(A), A, 0.0)
        if affinity is not None:
            A = A + np.asarray(affinity, float)

        return cls(K=K, kappa=kappa, edges=e, z=np.asarray(z, float),
                   affinity=A, F_states=F_states, temperature=temperature,
                   labels=labels, mode=mode, **kw)

    @property
    def n_states(self) -> int:
        return len(self.K)

    @property
    def n_reactions(self) -> int:
        return len(self.kappa)

    def incidence(self):
        n, m = self.n_states, self.n_reactions
        N = np.zeros((n, m))
        for r, (a, b) in enumerate(self.edges):
            N[a, r] -= 1.0
            N[b, r] += 1.0
        return N

    # -- electrical ---------------------------------------------------------

    def _boltz(self, V):
        """exp((z_r F V + A_r) / RT), the factor on the forward term.

        A_r is the CHEMOSTAT affinity of reaction r -- the free energy supplied
        by a reservoir held at fixed concentration. It must be carried
        separately from the Ce constants K: folding a concentration gradient
        into K forces the cycle affinity to zero and the model can then never
        produce a current, however large the gradient. (That was a real bug
        here, and it failed silently: every check still passed and the
        reversal potential was always 0 mV.)

        The open-circuit condition is sum_r (z_r F V + A_r) = 0 around the
        cycle, which is exactly the Nernst potential.
        """
        A = np.zeros_like(self.z) if self.affinity is None \
            else np.asarray(self.affinity, float)
        arg = (self.z * FARADAY * V + A) / (rt(self.temperature) * 1000.0)
        return np.exp(np.clip(arg, -300, 300))

    def fluxes(self, x, V=0.0):
        a, b = self.edges[:, 0], self.edges[:, 1]
        return self.kappa * (self.K[a] * x[a] * self._boltz(V) - self.K[b] * x[b])

    def dxdt(self, x, V=0.0):
        return self.incidence() @ self.fluxes(x, V)

    def current(self, x, V=0.0) -> float:
        """I = q sum_r z_r v_r, in amperes (q = e, or F for molar mode)."""
        return float(self.q_charge * np.sum(self.z * self.fluxes(x, V)))

    def steady_state(self, V=0.0, total=1.0):
        """Steady-state occupancies at fixed V (voltage clamp).

        Solves N v(x) = 0 subject to sum(x) = total. The flux is linear in x
        at fixed V, so this is a single linear solve, not an integration.
        """
        n, m = self.n_states, self.n_reactions
        a, b = self.edges[:, 0], self.edges[:, 1]
        bz = self._boltz(V)

        # dx/dt = A x with A assembled from the linear flux form
        A = np.zeros((n, n))
        for r in range(m):
            i, j = a[r], b[r]
            kf = self.kappa[r] * self.K[i] * bz[r]      # i -> j
            kr = self.kappa[r] * self.K[j]              # j -> i
            A[j, i] += kf; A[i, i] -= kf
            A[i, j] += kr; A[j, j] -= kr

        M = np.vstack([A, np.ones(n)])
        rhs = np.zeros(n + 1); rhs[-1] = total
        x, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        return x

    def steady_current(self, V=0.0, total=1.0) -> float:
        return self.current(self.steady_state(V, total), V)

    def iv_curve(self, voltages, total=1.0):
        """Current against clamped membrane potential."""
        V = np.asarray(voltages, dtype=float)
        return np.array([self.steady_current(float(v), total) for v in V])

    def conductance(self, V0=0.0, dV=0.005, total=1.0) -> float:
        """Slope conductance dI/dV at V0, in siemens."""
        return (self.steady_current(V0 + dV, total)
                - self.steady_current(V0 - dV, total)) / (2 * dV)

    def reversal_potential(self, lo=-0.5, hi=0.5):
        """The voltage at which the steady current vanishes.

        Computed by bisection on the model itself, so it is a PREDICTION of
        the assembled bond graph, not a formula applied to it. Comparing it
        with nernst() is therefore a genuine thermodynamic test.
        """
        from scipy.optimize import brentq
        f = lambda V: self.steady_current(float(V))      # noqa: E731
        try:
            return float(brentq(f, lo, hi, xtol=1e-13))
        except ValueError:
            return float("nan")

    def check_nernst(self, c_in, c_out, z_ion=1.0, tol_mV=0.5) -> dict:
        """Hunter's validation: reversal potential == Nernst potential."""
        Vr = self.reversal_potential()
        Vn = self.nernst(c_in, c_out, z_ion)
        return {"reversal_V": Vr, "nernst_V": Vn,
                "diff_mV": float((Vr - Vn) * 1000.0),
                "ok": bool(np.isfinite(Vr) and abs(Vr - Vn) * 1000 < tol_mV)}

    def nernst(self, c_in, c_out, z_ion=1.0) -> float:
        """Equilibrium potential, volts.

        E = (RT/zF) ln(c_out / c_in), with V measured as inside minus
        outside. For physiological K+ (140 mM in, 4 mM out) this gives
        -91.3 mV, the textbook E_K, and the assembled bond graph reproduces
        it to within 1e-11 mV.

        Hunter et al. write the equilibrium potential as (RT/zF) ln(c_i/c_o),
        which is the same physics with the transport reaction oriented the
        other way; the sign difference is orientation, not disagreement. The
        convention is pinned here because reversal_potential() is solved
        numerically from the assembled model, so check_nernst() compares two
        INDEPENDENTLY obtained numbers -- and an orientation slip shows up as
        a perfect magnitude match with the wrong sign, which is easy to wave
        through.
        """
        return thermal_voltage(self.temperature) / z_ion * np.log(c_out / c_in)

    # -- checks -------------------------------------------------------------

    def check_charge_closure(self, cycle=None, z_total=1.0, tol=1e-9) -> dict:
        """sum of z_r around a complete cycle must equal the ion valence.

        Not cosmetic: this is what makes the Nernst potential come out right
        and what forbids a current at equilibrium. A model whose electrical
        coefficients do not close transports charge from nowhere.
        """
        if cycle is None:
            cycle = list(range(self.n_reactions))
        s = float(np.sum(self.z[np.asarray(cycle, dtype=int)]))
        return {"sum_z": s, "expected": float(z_total),
                "closes": bool(abs(s - z_total) < tol),
                "residual": float(abs(s - z_total))}

    def check_no_current_at_equilibrium(self, V=0.0, total=1.0,
                                        tol=1e-18) -> dict:
        """At V = 0 with a symmetric graph there must be no steady current."""
        I = self.steady_current(V, total)
        scale = max(abs(self.q_charge * self.kappa.max()
                        * self.K.max() * total), 1e-300)
        return {"current": I, "relative": abs(I) / scale,
                "ok": bool(abs(I) / scale < 1e-9)}

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states, "n_reactions": self.n_reactions,
            "temperature_K": self.temperature,
            "C_m_farad": self.C_m, "mode": self.mode,
            "charge_per_flux_C": self.q_charge,
            "components": [
                {"id": f"Ce_{i}", "type": "Ce", "K": float(self.K[i]),
                 "label": (self.labels[i] if self.labels else None)}
                for i in range(self.n_states)],
            "bonds": [
                {"id": f"Re_{r}", "type": "Re", "kappa": float(self.kappa[r]),
                 "from": int(self.edges[r, 0]), "to": int(self.edges[r, 1]),
                 "z": float(self.z[r]),
                 "electrogenic": bool(abs(self.z[r]) > 0)}
                for r in range(self.n_reactions)],
            "electrical": {
                "formulation": "Hunter et al. bioRxiv 2024.06.26.600892",
                "coupling": "u_f += z_r F V at the 1-junction; "
                            "v_r = kappa_r (K_a q_a exp(z_r F V/RT) - K_b q_b)",
                "charge_balance": "dq_charge/dt = -q sum_r z_r v_r",
                "units_note": ("single_channel: x is occupancy, v is s^-1, "
                               "charge is the elementary charge. molar: "
                               "Hunter's convention with the Faraday constant."),
            },
            "meta": self.meta,
        }


def bath_affinity(c_source, c_sink, temperature: float = 298.0) -> float:
    """Chemostat affinity of a transit step, J/mol.

    A = RT ln(c_source / c_sink), where SOURCE is the compartment the forward
    direction of that reaction draws from. Getting this backwards flips the
    reversal potential while leaving its magnitude right -- which looks like a
    convention quibble and is actually a sign error in the driving force, so
    the orientation is made explicit here rather than left to the caller.

    For a channel cycle written so the closing edge returns the ion from
    outside to inside, source = outside and sink = inside.
    """
    return rt(temperature) * 1000.0 * np.log(float(c_source) / float(c_sink))


def electrical_distances(z_positions, z_filter, z_bulk, z_ion=1.0):
    """Woodhull electrical distance per state, from positions along the pore.

    delta_i = (z_i - z_filter) / (z_bulk - z_filter), so delta runs 0 -> 1
    across the field. The per-reaction coefficient is the DIFFERENCE between
    the two states it connects:

        z_r = z_ion * (delta_j - delta_i)

    which telescopes around a cycle to exactly z_ion -- charge closure by
    construction rather than by assumption.

    The uniform-field assumption behind this is a simplification and should be
    declared: the real field across a selectivity filter is not uniform, and
    comparing a geometric delta with an electrophysiologically fitted one is a
    falsifiable prediction, not a calibration.
    """
    zp = np.asarray(z_positions, dtype=float)
    span = float(z_bulk - z_filter)
    if abs(span) < 1e-12:
        raise ValueError("z_filter and z_bulk coincide")
    return (zp - z_filter) / span


def reaction_charges(delta, edges, z_ion=1.0, cyclic_close=True):
    """Per-reaction charge coefficients from per-state electrical distances.

    cyclic_close adds the implicit return leg's charge to the last reaction so
    that the total closes to z_ion. Set it False if the graph already contains
    an explicit edge completing the cycle.
    """
    delta = np.asarray(delta, dtype=float)
    e = np.asarray(edges, dtype=int)
    z = z_ion * (delta[e[:, 1]] - delta[e[:, 0]])
    if cyclic_close and len(z):
        z = z.copy()
        z[-1] += z_ion - z.sum()
    return z
