"""Dimension-general estimation. This is where the pipeline actually scales.

THE SCALING PROBLEM
-------------------
The grid route costs n^d bins: 60^2 = 3,600 is fine, 60^4 = 12.96 million is
not, and it gets worse than the memory cost suggests -- the number of SAMPLED
bins stops growing once it passes the number of frames, so beyond ~3 CVs almost
every bin is empty, the generator disconnects, and (as the regression tests
pin) a disconnected generator silently deletes the slow process.

MSMs hit the same wall from the other side: microstate counts must grow with
dimension to keep resolution, while the counts per state fall.

THE FIX, AND WHY THE PHNN IS THE SCALABLE PIECE
-----------------------------------------------
Nothing requires us to enumerate bins. Hash each frame to an integer voxel key
and keep only OCCUPIED voxels. The cost is then O(n_frames), independent of
dimension, and the number of fit targets is bounded by the data rather than by
n^d. The learned H and D are continuous functions, so they are defined
everywhere without any bin ever existing.

The grid is still used for the BOND GRAPH, because a bond graph is a discrete
object and needs discrete states -- but it is built from basins (a handful of
states), not from n^d bins. So the expensive part scales and the discrete part
stays small.
"""

from __future__ import annotations

import numpy as np

from .cvs import wrapped_delta

TWO_PI = 2.0 * np.pi


def voxel_keys(X: np.ndarray, n_per_dim: int):
    """Integer voxel index per frame, for periodic coordinates in [-pi, pi).

    Returns (keys, idx_per_dim). Only occupied voxels ever materialise, so the
    memory cost is O(n_frames) and does not depend on dimension.
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    h = TWO_PI / n_per_dim
    idx = np.floor((X + np.pi) / h).astype(np.int64) % n_per_dim
    # mixed-radix encode; int64 holds n_per_dim**d for sensible n and d
    key = np.zeros(len(X), dtype=np.int64)
    for k in range(d):
        key = key * n_per_dim + idx[:, k]
    return key, idx


def sparse_binned_drift(X, dt, n_per_dim=20, traj_id=None, lag=1, min_count=20):
    """First Kramers-Moyal moment on OCCUPIED voxels only, in any dimension.

    Identical in spirit to phnn.binned_drift, but it never allocates n^d
    anything. Returns the voxel centres, the mean drift there, and the counts.
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)

    ok = traj_id[:-lag] == traj_id[lag:]
    x0 = X[:-lag][ok]
    dx = wrapped_delta(X[lag:][ok], X[:-lag][ok])
    tau = dt * lag

    key, idx = voxel_keys(x0, n_per_dim)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)

    sums = np.zeros((len(uniq), d))
    for k in range(d):
        sums[:, k] = np.bincount(inv, weights=dx[:, k], minlength=len(uniq))
    drift = sums / (counts[:, None] * tau)

    # voxel centre from the first member of each voxel
    first = np.zeros(len(uniq), dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    h = TWO_PI / n_per_dim
    centers = (idx[first].astype(np.float64) + 0.5) * h - np.pi

    keep = counts >= min_count
    return {
        "centers": centers[keep], "drift": drift[keep], "counts": counts[keep],
        "n_occupied": int(len(uniq)), "n_kept": int(keep.sum()),
        "dim": d, "n_per_dim": n_per_dim,
        "n_bins_if_dense": float(n_per_dim) ** d,
    }


def measure_D_nd(X, dt, traj_id=None, lag=1):
    """Isotropic D in any dimension, from the second moment."""
    X = np.asarray(X, dtype=np.float64)
    if traj_id is None:
        traj_id = np.zeros(len(X), dtype=int)
    traj_id = np.asarray(traj_id)
    ok = traj_id[:-lag] == traj_id[lag:]
    dx = wrapped_delta(X[lag:][ok], X[:-lag][ok])
    per_dim = np.mean(dx ** 2, axis=0) / (2.0 * dt * lag)
    return {"D_per_dim": per_dim, "D_const": float(np.mean(per_dim))}


def train_nd(X, dt, D_const, dim=None, n_per_dim=20, temperature=298.0,
             traj_id=None, lag=1, min_count=20, steps=3000, lr=1e-2,
             seed=0, scale=10.0, weight_decay=1e-4, patience=600,
             val_frac=0.2, verbose=False, device=None, n_harmonics=2,
             width=64, depth=2):
    """Fit H in arbitrary dimension against the sparse binned drift.

    Same estimator as the 2D path, same early stopping, no dense grid.
    """
    import torch

    from .phnn import FreeEnergyNet, _drift_structured
    from .thermo.smoluchowski import rt

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dim = dim or np.asarray(X).shape[1]

    bd = sparse_binned_drift(X, dt, n_per_dim, traj_id, lag, min_count)
    pts, tgt, w = bd["centers"], bd["drift"], bd["counts"].astype(np.float64)
    n = len(pts)
    if n < 10:
        raise ValueError(f"only {n} voxels survive min_count={min_count}")

    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    vi, ti = perm[:n_val], perm[n_val:]

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    x_tr, y_tr, w_tr = T(pts[ti]), T(tgt[ti]), T(w[ti])
    x_va, y_va, w_va = T(pts[vi]), T(tgt[vi]), T(w[vi])

    # n_harmonics MUST cover the oscillation of the target. The embedding is
    # sin/cos up to n_harmonics, so a landscape with p periods across the
    # window is INEXPRESSIBLE below n_harmonics = p. The KcsA filter has ~5.5
    # site periods across an 18 A window; at the default 2 the network cannot
    # represent it at all and returns a nearly flat H however long it trains.
    model = FreeEnergyNet(dim=dim, n_harmonics=n_harmonics, scale=scale,
                          width=width, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    beta = 1.0 / rt(temperature)

    def loss_on(x, y, ww):
        pred = _drift_structured(model, x, D_const, beta)
        return (ww * ((pred - y) ** 2).sum(dim=-1)).sum() / ww.sum()

    best = {"val": float("inf"), "step": -1, "state": None}
    history = []
    for step in range(steps):
        opt.zero_grad()
        loss = loss_on(x_tr, y_tr, w_tr)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.enable_grad():
                vl = float(loss_on(x_va, y_va, w_va).detach())
            if vl < best["val"]:
                best = {"val": vl, "step": step,
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}}
            if step % 250 == 0 and verbose:
                print(f"  step {step:5d} train {float(loss.detach()):.6f} val {vl:.6f}")
            if patience and step - best["step"] >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    wt = np.asarray(w)
    var = float(np.sum(wt * (tgt ** 2).sum(axis=1)) / wt.sum())
    return model, history, {
        "dim": dim, "n_voxels_fit": n, "n_occupied": bd["n_occupied"],
        "n_bins_if_dense": bd["n_bins_if_dense"],
        "sparsity": bd["n_occupied"] / max(bd["n_bins_if_dense"], 1.0),
        "best_step": best["step"], "best_val": best["val"],
        "r2": 1.0 - best["val"] / max(var, 1e-30),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
