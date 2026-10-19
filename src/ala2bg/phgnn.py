"""Port-Hamiltonian graph neural network on the minimal state graph.

WHAT IT IS FOR
--------------
Once the dynamics has chosen the states (minimal.py), the bond graph is a small
labelled graph: nodes carry Ce constants K_i, edges carry Re constants kappa_ij.
For a system you have already sampled, you can read both straight off the
measured rate matrix and a network adds nothing.

The network earns its place in exactly one way: TRANSFERABILITY. Predicting
kappa for a state pair, a mutant or a homologue you have NOT sampled, from
features you can compute cheaply.

That is the useful asymmetry in practice: state free energies are cheap now
(generative ensembles, continuum electrostatics, structures), while rate
constants need a path-sampling campaign. So the task here is deliberately:

    given cheap node features (state free energies + geometry)
    predict the expensive edge parameter kappa

THE STRUCTURE IS ARCHITECTURAL, NOT FITTED
------------------------------------------
Two invariants hold for ANY weights, which is what makes this a
port-Hamiltonian network rather than a network with physics bolted on:

1. kappa_ij is SYMMETRIC, because the edge readout consumes only symmetric
   functions of its two endpoints -- (h_i + h_j) and |h_i - h_j|. Swapping i
   and j cannot change the output. Symmetry of kappa IS detailed balance (the
   same guarantee Bicout-Szabo gives on a grid and symmetrised counts give on
   an MSM), so the Wegscheider cycle condition holds by construction and no
   cycle-closure penalty is ever needed.

2. K_i and kappa_ij are positive, because the readouts are exponentials of the
   network output. So mu_i = RT ln(K_i x_i) is always defined and R is PSD.

The consequence worth stating plainly: a free-form model of the same graph has
one free parameter per ORDERED pair, and must be penalised towards detailed
balance. This one has a function of symmetric features and satisfies it
exactly, at any point in training.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False
    torch, nn = None, object

from .thermo.smoluchowski import rt


# ---------------------------------------------------------------------------
# Synthetic graph dataset: random periodic landscapes with EXACT coarse rates
# ---------------------------------------------------------------------------

def random_landscape(grid, rng, n_modes=3, amp=12.0):
    """A random smooth periodic landscape on the torus, in kJ/mol."""
    pts = grid.center_grid()
    F = np.zeros(len(pts))
    for k in range(1, n_modes + 1):
        F += (rng.normal(0, amp / k) * np.cos(k * pts[:, 0] + rng.uniform(0, 2 * np.pi))
              + rng.normal(0, amp / k) * np.cos(k * pts[:, 1] + rng.uniform(0, 2 * np.pi)))
    F += 0.4 * amp * rng.normal() * np.cos(pts[:, 0] + rng.uniform(0, 2 * np.pi)) \
        * np.cos(pts[:, 1] + rng.uniform(0, 2 * np.pi))
    return F - F.min()


def basins_of(F, grid, max_states=6):
    """Local minima of F on the periodic grid, as state centres."""
    n1, n2 = grid.shape
    Fg = F.reshape(n1, n2)
    is_min = np.ones_like(Fg, dtype=bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            is_min &= Fg <= np.roll(np.roll(Fg, a, 0), b, 1)
    idx = np.flatnonzero(is_min.ravel())
    if len(idx) > max_states:                 # keep the deepest
        idx = idx[np.argsort(F[idx])[:max_states]]
    return idx


def exact_graph(grid, F, D, temperature=298.0, max_states=6):
    """Build a state graph with EXACT rates from the discretised generator.

    Offline data generation only -- this uses the grid route deliberately,
    because here we want ground truth, not scalability.
    """
    from .thermo import committor as cm
    from .thermo.smoluchowski import build_generator

    L, keep = build_generator(F, D, grid, temperature)
    remap = -np.ones(grid.n_bins, dtype=int)
    remap[keep] = np.arange(len(keep))

    centres = basins_of(F, grid, max_states)
    centres = [c for c in centres if remap[c] >= 0]
    n = len(centres)
    if n < 2:
        return None

    pts = grid.center_grid()
    beta = 1.0 / rt(temperature)

    # assign every kept bin to its nearest basin on the torus
    cpts = pts[centres]
    d = np.stack([
        np.linalg.norm((pts[keep] - c + np.pi) % (2 * np.pi) - np.pi, axis=1)
        for c in cpts], axis=1)
    lab = np.argmin(d, axis=1)

    # state free energies from summed Boltzmann weight
    w = np.exp(-beta * (F[keep] - F[keep].min()))
    Fs = np.array([-rt(temperature) * np.log(max(w[lab == s].sum(), 1e-300))
                   for s in range(n)])
    Fs -= Fs.min()

    # exact pairwise rates by MFPT on the generator
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A = np.flatnonzero(lab == i)
            B = np.flatnonzero(lab == j)
            if not A.size or not B.size:
                continue
            pi = np.exp(-beta * (F[keep] - F[keep].min()))
            pi /= pi.sum()
            try:
                t = cm.mfpt_between(L, A, B, pi)
            except Exception:
                t = np.nan
            if np.isfinite(t) and t > 0:
                K[i, j] = 1.0 / t

    # enforce detailed balance on the coarse rates
    piS = np.exp(-beta * Fs); piS /= piS.sum()
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > 0 and K[j, i] > 0:
                f = 0.5 * (piS[i] * K[i, j] + piS[j] * K[j, i])
                K[i, j] = f / piS[i]
                K[j, i] = f / piS[j]

    Kc = np.exp(beta * Fs) / grid.cell_area
    edges, kappa = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > 0 and K[j, i] > 0:
                edges.append((i, j))
                kappa.append(K[i, j] / Kc[i])
    if not edges:
        return None

    # per-edge saddle: the feature that actually governs kappa
    from .barrier import minimax_barrier
    sad = minimax_barrier(F, grid, [int(c) for c in centres])
    E = np.array(edges)
    saddle = np.array([sad[i, j] for i, j in E])

    return {
        "centres": pts[centres], "F": Fs, "K_ce": Kc,
        "edges": E, "kappa": np.array(kappa),
        "D": D, "rate_matrix": K, "saddle": saddle,
        "grid_cell_area": grid.cell_area, "grid_h": float(grid.h[0]),
    }


def make_dataset(n_graphs=120, n_grid=40, seed=0, temperature=298.0):
    """An ensemble of small state graphs with exact kappa."""
    from .grid import TorusGrid
    rng = np.random.default_rng(seed)
    grid = TorusGrid(n_grid, n_grid)
    out = []
    for g in range(n_graphs):
        F = random_landscape(grid, rng, n_modes=int(rng.integers(2, 4)),
                             amp=float(rng.uniform(6, 18)))
        D = float(10 ** rng.uniform(-2.3, -1.7))
        try:
            rec = exact_graph(grid, F, D, temperature)
        except Exception:
            rec = None
        if rec and len(rec["edges"]) >= 1:
            rec["graph_id"] = g
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Features -- cheap ones only
# ---------------------------------------------------------------------------

def node_features(rec):
    """Cheap per-state features: position on the torus and state free energy.

    Deliberately excludes anything that encodes the barrier, because the
    barrier is the expensive thing we are trying to avoid computing.
    """
    c = rec["centres"]
    F = rec["F"]
    return np.concatenate([
        np.sin(c), np.cos(c),
        F[:, None] / 10.0,
        np.full((len(F), 1), np.log10(rec["D"])),
    ], axis=1)


def edge_geometry(rec, use_barrier=False):
    """Symmetric edge features. Every column is invariant under swapping the
    two endpoints, which is what keeps kappa symmetric by construction.

    use_barrier adds the saddle descriptors. Those are the features that can
    actually inform kappa (ln kappa ~ ln D - beta F_dagger); without them the
    network is being asked to predict a barrier it has no information about.
    """
    c = rec["centres"]
    e = rec["edges"]
    dv = (c[e[:, 1]] - c[e[:, 0]] + np.pi) % (2 * np.pi) - np.pi
    dist = np.linalg.norm(dv, axis=1, keepdims=True)
    feats = [dist, np.abs(dv)]                       # |dv| is symmetric

    if use_barrier:
        sad = np.asarray(rec["saddle"], dtype=float)
        Fi, Fj = rec["F"][e[:, 0]], rec["F"][e[:, 1]]
        feats += [
            (sad / 10.0)[:, None],                              # F_dagger
            ((sad - np.minimum(Fi, Fj)) / 10.0)[:, None],       # from deeper well
            ((sad - np.maximum(Fi, Fj)) / 10.0)[:, None],       # from shallower
        ]
    out = np.concatenate(feats, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class PHGNN(nn.Module):
    """Message passing -> (log K_i, log kappa_ij), with kappa symmetric by
    construction."""

    def __init__(self, n_node_feat, n_edge_feat, hidden=48, rounds=2):
        if not HAVE_TORCH:
            raise ImportError("PyTorch required")
        super().__init__()
        self.rounds = rounds
        self.embed = nn.Sequential(nn.Linear(n_node_feat, hidden), nn.SiLU())
        # message depends on symmetric endpoint features + edge geometry
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + n_edge_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
        self.node_out = nn.Linear(hidden, 1)
        self.edge_out = nn.Sequential(
            nn.Linear(2 * hidden + n_edge_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    @staticmethod
    def _sym(hi, hj):
        """The symmetry guarantee lives here, and nowhere else.

        Only (h_i + h_j) and |h_i - h_j| are exposed downstream, and both are
        invariant under swapping i and j. Therefore kappa_ij == kappa_ji
        identically, for any weights -- detailed balance is structural.
        """
        return torch.cat([hi + hj, torch.abs(hi - hj)], dim=-1)

    def forward(self, x, edge_index, edge_attr):
        h = self.embed(x)
        src, dst = edge_index[:, 0], edge_index[:, 1]
        for _ in range(self.rounds):
            m = self.msg(torch.cat([self._sym(h[src], h[dst]), edge_attr], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, src, m)
            agg.index_add_(0, dst, m)          # symmetric aggregation
            h = self.upd(torch.cat([h, agg], dim=-1))
        log_K = self.node_out(h).squeeze(-1)
        log_kappa = self.edge_out(
            torch.cat([self._sym(h[src], h[dst]), edge_attr], dim=-1)).squeeze(-1)
        return log_K, log_kappa


def _pack(rec, device, use_barrier=False):
    x = torch.tensor(node_features(rec), dtype=torch.float32, device=device)
    ei = torch.tensor(rec["edges"], dtype=torch.long, device=device)
    ea = torch.tensor(edge_geometry(rec, use_barrier),
                      dtype=torch.float32, device=device)
    y = torch.tensor(np.log(np.clip(rec["kappa"], 1e-300, None)),
                     dtype=torch.float32, device=device)
    return x, ei, ea, y


def train(dataset, steps=1500, lr=3e-3, hidden=48, rounds=2, seed=0,
          val_frac=0.25, device=None, verbose=False, use_barrier=False):
    """Train on whole graphs; hold out whole graphs for the transfer test."""
    if not HAVE_TORCH:
        raise ImportError("PyTorch required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or "cpu"

    idx = rng.permutation(len(dataset))
    n_val = max(1, int(round(val_frac * len(dataset))))
    val = [dataset[i] for i in idx[:n_val]]
    trn = [dataset[i] for i in idx[n_val:]]

    nf = node_features(dataset[0]).shape[1]
    ef = edge_geometry(dataset[0], use_barrier).shape[1]
    model = PHGNN(nf, ef, hidden, rounds).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    packed_t = [_pack(r, device, use_barrier) for r in trn]
    packed_v = [_pack(r, device, use_barrier) for r in val]

    def loss_on(packs):
        tot, n = 0.0, 0
        for x, ei, ea, y in packs:
            _, lk = model(x, ei, ea)
            tot = tot + ((lk - y) ** 2).sum()
            n += len(y)
        return tot / max(n, 1)

    best = {"val": float("inf"), "state": None, "step": -1}
    for s in range(steps):
        opt.zero_grad()
        l = loss_on(packed_t)
        l.backward()
        opt.step()
        if s % 25 == 0 or s == steps - 1:
            with torch.no_grad():
                v = float(loss_on(packed_v))
            if v < best["val"]:
                best = {"val": v, "step": s,
                        "state": {k: t.detach().clone()
                                  for k, t in model.state_dict().items()}}
            if verbose and s % 250 == 0:
                print(f"  step {s:5d}  train {float(l):.4f}  val {v:.4f}")
    if best["state"]:
        model.load_state_dict(best["state"])
    model.eval()
    return model, trn, val, best


def evaluate(model, graphs, device="cpu", use_barrier=False):
    """RMSE in ln kappa on held-out graphs, plus the naive baseline.

    Baseline: predict ln kappa from the endpoint free energies alone, which is
    what you would do with the cheap parameters and no model of the barrier.
    If the network cannot beat that, it is not earning its place.
    """
    pred, true, base_feat = [], [], []
    for r in graphs:
        x, ei, ea, y = _pack(r, device, use_barrier)
        with torch.no_grad():
            _, lk = model(x, ei, ea)
        pred.append(lk.cpu().numpy())
        true.append(y.cpu().numpy())
        e = r["edges"]
        base_feat.append(np.stack([
            r["F"][e[:, 0]] + r["F"][e[:, 1]],
            np.abs(r["F"][e[:, 0]] - r["F"][e[:, 1]]),
            np.full(len(e), np.log10(r["D"])),
        ], axis=1))
    p = np.concatenate(pred); t = np.concatenate(true)
    B = np.concatenate(base_feat)

    A = np.concatenate([B, np.ones((len(B), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    b = A @ coef

    def rmse(a):
        return float(np.sqrt(np.mean((a - t) ** 2)))

    return {
        "n_edges": int(len(t)), "n_graphs": len(graphs),
        "rmse_phgnn_ln_kappa": rmse(p),
        "rmse_baseline_ln_kappa": rmse(b),
        "rmse_mean_predictor": rmse(np.full_like(t, t.mean())),
        "improvement_over_baseline": float(rmse(b) / max(rmse(p), 1e-12)),
        "corr_phgnn": float(np.corrcoef(p, t)[0, 1]) if len(t) > 2 else float("nan"),
    }


def check_symmetry(model, rec, device="cpu"):
    """kappa_ij must equal kappa_ji identically -- swap every edge and compare.

    This is the structural claim, so it is checked numerically rather than
    asserted in a docstring.
    """
    x, ei, ea, _ = _pack(rec, device)
    with torch.no_grad():
        _, a = model(x, ei, ea)
        _, b = model(x, ei.flip(1), ea)
    # RELATIVE error. An absolute threshold is wrong here: log kappa can be
    # O(1e5) for a badly-scaled or perturbed network, where float32 round-off
    # alone is O(0.1) -- which would look like a symmetry violation and is not.
    scale = float(torch.maximum(a.abs(), b.abs()).max().clamp(min=1.0))
    return float((a - b).abs().max()) / scale
