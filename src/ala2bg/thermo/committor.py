"""Committor and mean first passage time -- exactly, by sparse linear solve.

This is the Gen-COMPAS substitute. Gen-COMPAS learns the committor with a
Variational Committor Network because its state space is 3N Cartesian
coordinates. Ours is a 2-CV grid, so the committor has an exact answer on the
same generator that becomes the bond graph, in milliseconds:

    committor   L^T q = 0,   q = 0 on A, q = 1 on B
    MFPT to B   L^T tau = -1, tau = 0 on B

No targeted MD, no NAMD, no VMD, no seed trajectories, no convergence guesswork.
The transition-state ensemble is the q = 0.5 isoband -- exactly the criterion
Gen-COMPAS uses to pick its targets.

Note on convention: build_generator returns L with dp/dt = L p, so L[j, i] is
the rate i->j. The backward (Kolmogorov) operator acting on functions of the
state is therefore Q = L.T, with Q[i, j] = k_{i->j}.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _backward(L):
    return sp.csr_matrix(L).T.tocsr()


def committor(L, A_idx, B_idx) -> np.ndarray:
    """Forward committor q: probability of reaching B before A.

    A_idx, B_idx are index arrays into the generator's state space.
    """
    Q = _backward(L)
    n = Q.shape[0]
    A = np.zeros(n, dtype=bool); A[np.asarray(A_idx, dtype=int)] = True
    B = np.zeros(n, dtype=bool); B[np.asarray(B_idx, dtype=int)] = True
    if (A & B).any():
        raise ValueError("A and B overlap")
    T = ~(A | B)
    if not T.any():
        raise ValueError("no transient states")

    q = np.zeros(n)
    q[B] = 1.0
    # Q_TT q_T = -Q_TB q_B
    Q_TT = Q[T][:, T]
    rhs = -np.asarray(Q[T][:, B].sum(axis=1)).ravel()
    q[T] = spl.spsolve(Q_TT.tocsc(), rhs)
    return q


def mfpt(L, target_idx) -> np.ndarray:
    """Mean first passage time to the target set, in the generator's time unit.

    Exact on the discretised generator -- no Kramers high-barrier approximation
    anywhere. That matters because Kramers is quantitatively poor for the
    modest barriers alanine dipeptide actually has.
    """
    Q = _backward(L)
    n = Q.shape[0]
    B = np.zeros(n, dtype=bool); B[np.asarray(target_idx, dtype=int)] = True
    T = ~B
    if not T.any():
        raise ValueError("target is everything")

    tau = np.zeros(n)
    Q_TT = Q[T][:, T]
    tau[T] = spl.spsolve(Q_TT.tocsc(), -np.ones(T.sum()))
    return tau


def mfpt_between(L, A_idx, B_idx, pi) -> float:
    """pi-weighted MFPT from set A to set B."""
    tau = mfpt(L, B_idx)
    A = np.asarray(A_idx, dtype=int)
    w = np.asarray(pi)[A]
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(tau[A] * w) / w.sum())


def timescales(L, k: int = 5):
    """Relaxation timescales -1/lambda_i from the generator's spectrum.

    lambda_0 = 0 (stationary); the rest are negative. Returned largest first,
    in the generator's time unit.
    """
    n = L.shape[0]
    k_eff = min(k + 1, n - 2)
    if k_eff < 1:
        return np.array([])
    try:
        vals = spl.eigs(sp.csr_matrix(L).astype(np.float64), k=k_eff,
                        which="LR", return_eigenvectors=False)
        lam = np.sort(np.real(vals))[::-1]
    except Exception:
        lam = np.sort(np.real(np.linalg.eigvals(np.asarray(L.todense()))))[::-1]
    lam = lam[lam < -1e-14]
    if lam.size == 0:
        return np.array([])
    return np.sort(-1.0 / lam)[::-1][:k]
