"""Generator -> port-Hamiltonian bond graph (Gawthrop-Crampin formalism).

Effort = chemical potential mu [kJ/mol]; flow = molar rate v; power = mu * v.

Ce (chemical capacitor), one per state:

    mu_i = RT ln(K_i x_i),      K_i = exp(beta F_i) / V_i

Setting K_i from the landscape IS the mapping: equal potentials then give
x_i / x_j = exp(-beta (F_i - F_j)), the Boltzmann distribution, by construction.

Re (reaction resistor, two-port), one per edge, Marcelin-de Donder:

    v_ij = kappa_ij ( exp(mu_i/RT) - exp(mu_j/RT) ) = kappa_ij ( K_i x_i - K_j x_j )

Matching the Smoluchowski rates requires kappa_ij K_i = k_{i->j}, giving

    kappa_ij = (D_ij V_i / h^2) exp( -beta (F_i + F_j) / 2 )

SYMMETRY OF kappa_ij IN (i, j) *IS* THERMODYNAMIC CONSISTENCY -- the same
resistor seen from both ports. Detailed balance is structural, not fitted.

Port-Hamiltonian form, with N the incidence matrix of the Re edges:

    H(x) = sum_i RT x_i ( ln(K_i x_i) - 1 ),    dH/dx = mu
    xdot = N v = -R(x) dH/dx,   R(x) = N diag(g) N^T >= 0,   Hdot = -mu^T R mu <= 0

J = 0, which is correct for overdamped dynamics where momenta have been
integrated out. R is nonlinear in the effort and we say so rather than
claiming a linear-resistive form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .smoluchowski import rt


@dataclass
class BondGraph:
    """Provider-neutral intermediate representation."""

    K: np.ndarray                      # (n,) Ce constants
    kappa: np.ndarray                  # (m,) Re constants
    edges: np.ndarray                  # (m, 2) int, endpoints
    F: np.ndarray                      # (n,) free energy kJ/mol
    temperature: float = 298.0
    volumes: np.ndarray | None = None
    labels: list | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.K)

    @property
    def n_reactions(self) -> int:
        return len(self.kappa)

    def incidence(self) -> sp.csr_matrix:
        """N, (n_states, n_reactions). Column r has -1 at tail, +1 at head."""
        n, m = self.n_states, self.n_reactions
        rows, cols, vals = [], [], []
        for r, (a, b) in enumerate(self.edges):
            rows += [a, b]; cols += [r, r]; vals += [-1.0, 1.0]
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, m)).tocsr()

    def mu(self, x: np.ndarray) -> np.ndarray:
        return rt(self.temperature) * np.log(np.clip(self.K * x, 1e-300, None))

    def fluxes(self, x: np.ndarray) -> np.ndarray:
        a, b = self.edges[:, 0], self.edges[:, 1]
        return self.kappa * (self.K[a] * x[a] - self.K[b] * x[b])

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        return self.incidence() @ self.fluxes(x)

    def hamiltonian(self, x: np.ndarray) -> float:
        xs = np.clip(x, 1e-300, None)
        return float(np.sum(rt(self.temperature) * xs * (np.log(self.K * xs) - 1.0)))

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "n_reactions": self.n_reactions,
            "temperature_K": self.temperature,
            "RT_kJ_per_mol": rt(self.temperature),
            "components": [
                {"id": f"Ce_{i}", "type": "Ce", "K": float(self.K[i]),
                 "F_kJ_per_mol": float(self.F[i]),
                 "label": (self.labels[i] if self.labels else None)}
                for i in range(self.n_states)
            ],
            "bonds": [
                {"id": f"Re_{r}", "type": "Re", "kappa": float(self.kappa[r]),
                 "from": int(self.edges[r, 0]), "to": int(self.edges[r, 1])}
                for r in range(self.n_reactions)
            ],
            "meta": self.meta,
        }


def from_generator(L, F_keep, grid_cell_area: float, temperature: float = 298.0,
                   labels=None, meta=None) -> BondGraph:
    """Build the bond graph from a generator and the free energies on its states.

    kappa_ij = k_{i->j} / K_i is computed from the generator, so whatever
    produced L (PHNN-learned H, or the empirical histogram FES) flows through
    unchanged. The symmetry assertion below is the thermodynamic-consistency
    check and it is not optional.
    """
    F = np.asarray(F_keep, dtype=np.float64).ravel()
    n = F.size
    beta = 1.0 / rt(temperature)
    V = grid_cell_area
    K = np.exp(beta * (F - F.min())) / V

    Lc = sp.coo_matrix(L)
    rate = {}
    for i, j, v in zip(Lc.row, Lc.col, Lc.data):
        if i != j:
            rate[(int(j), int(i))] = float(v)   # L[i, j] = k_{j->i}

    seen, edges, kappas = set(), [], []
    for (a, b), k_ab in rate.items():
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        kap = k_ab / K[a]
        k_ba = rate.get((b, a))
        if k_ba is not None:
            kap_rev = k_ba / K[b]
            rel = abs(kap - kap_rev) / max(abs(kap), abs(kap_rev), 1e-300)
            if rel > 1e-8:
                raise AssertionError(
                    f"kappa not symmetric on edge ({a},{b}): "
                    f"{kap:.6e} vs {kap_rev:.6e} (rel {rel:.2e}). "
                    "Detailed balance is broken upstream."
                )
            kap = 0.5 * (kap + kap_rev)
        edges.append((a, b))
        kappas.append(kap)

    return BondGraph(
        K=K, kappa=np.array(kappas), edges=np.array(edges, dtype=int),
        F=F, temperature=temperature,
        volumes=np.full(n, V), labels=labels, meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Cycle tests
# ---------------------------------------------------------------------------

def cycle_residual(k_fwd: np.ndarray, k_rev: np.ndarray) -> float:
    """|prod(k_fwd)/prod(k_rev) - 1| for one cycle, computed in log space."""
    lf = np.sum(np.log(np.clip(k_fwd, 1e-300, None)))
    lr = np.sum(np.log(np.clip(k_rev, 1e-300, None)))
    return float(abs(np.expm1(lf - lr)))


def wegscheider_residuals(L, grid, keep, n_random: int = 50, seed: int = 0) -> dict:
    """Cycle residuals for contractible plaquettes AND non-contractible loops.

    The grid is a torus, so it admits cycles that wind around phi or psi and
    cannot be contracted to a point. At equilibrium the net affinity around ANY
    cycle must vanish.

    HONESTY NOTE, and it matters: for a model whose drift is -D grad H with H a
    single-valued periodic function, these residuals vanish AUTOMATICALLY --
    the affinities telescope. Checked alone, this test is near-vacuous and
    claiming it as a triumph invites a deserved objection. It acquires teeth
    only by comparison with the unconstrained baseline (see phnn.py), which is
    under no such obligation and does violate it. Report both, side by side.
    """
    Lc = sp.coo_matrix(L)
    rate = {}
    for i, j, v in zip(Lc.row, Lc.col, Lc.data):
        if i != j:
            rate[(int(j), int(i))] = float(v)

    remap = -np.ones(grid.n_bins, dtype=int)
    remap[keep] = np.arange(len(keep))

    def k(a, b):
        return rate.get((a, b))

    def walk(flat_cycle):
        kf, kr = [], []
        for a, b in zip(flat_cycle, flat_cycle[1:] + flat_cycle[:1]):
            ka, kb = k(a, b), k(b, a)
            if ka is None or kb is None:
                return None
            kf.append(ka); kr.append(kb)
        return cycle_residual(np.array(kf), np.array(kr))

    rng = np.random.default_rng(seed)
    out = {"plaquette": [], "winding_phi": [], "winding_psi": []}

    # contractible unit plaquettes
    for _ in range(n_random):
        i = int(rng.integers(grid.n_phi)); j = int(rng.integers(grid.n_psi))
        ip, jp = (i + 1) % grid.n_phi, (j + 1) % grid.n_psi
        cyc = [remap[grid.ravel(i, j)], remap[grid.ravel(ip, j)],
               remap[grid.ravel(ip, jp)], remap[grid.ravel(i, jp)]]
        if min(cyc) < 0:
            continue
        r = walk([int(c) for c in cyc])
        if r is not None:
            out["plaquette"].append(r)

    # non-contractible loops: all the way round phi (at fixed psi) and vice versa
    for j in range(grid.n_psi):
        cyc = [remap[grid.ravel(i, j)] for i in range(grid.n_phi)]
        if min(cyc) < 0:
            continue
        r = walk([int(c) for c in cyc])
        if r is not None:
            out["winding_phi"].append(r)
    for i in range(grid.n_phi):
        cyc = [remap[grid.ravel(i, j)] for j in range(grid.n_psi)]
        if min(cyc) < 0:
            continue
        r = walk([int(c) for c in cyc])
        if r is not None:
            out["winding_psi"].append(r)

    return {
        key: {
            "n": len(v),
            "max": float(np.max(v)) if v else None,
            "median": float(np.median(v)) if v else None,
        }
        for key, v in out.items()
    }


def check_thermo(bg: BondGraph, pi: np.ndarray | None = None) -> dict:
    """Structural checks that go straight onto the slide."""
    N = bg.incidence()
    beta = 1.0 / rt(bg.temperature)

    x_eq = np.exp(-beta * (bg.F - bg.F.min()))
    x_eq /= x_eq.sum()
    v_eq = bg.fluxes(x_eq)
    dx_eq = np.abs(N @ v_eq)

    # Reference scale for "is this flux zero?": the magnitude of the individual
    # one-way terms whose difference the flux is. At equilibrium the difference
    # cancels to machine precision while each term stays O(1), so normalising
    # by the terms is the meaningful relative test. Normalising by the net flux
    # itself would be 0/0.
    a_e, b_e = bg.edges[:, 0], bg.edges[:, 1]
    oneway = bg.kappa * np.maximum(bg.K[a_e] * x_eq[a_e], bg.K[b_e] * x_eq[b_e])
    scale = max(float(np.max(oneway)), 1e-300)

    rng = np.random.default_rng(0)
    x = rng.random(bg.n_states) + 0.1
    x /= x.sum()
    mu = bg.mu(x)
    v = bg.fluxes(x)
    A = -(N.T @ mu)                 # forward affinity per reaction
    dissipation = float(np.sum(A * v))
    Hdot = float(mu @ (N @ v))

    return {
        "equilibrium_flux_max": float(dx_eq.max()),
        "equilibrium_flux_rel": float(dx_eq.max() / scale),
        "equilibrium_flux_scale": scale,
        "dissipation_nonneg": bool(dissipation >= -1e-9),
        "dissipation": dissipation,
        "Hdot": Hdot,
        "Hdot_nonpos": bool(Hdot <= 1e-9),
        "kappa_min": float(bg.kappa.min()),
        "kappa_all_positive": bool((bg.kappa > 0).all()),
    }
