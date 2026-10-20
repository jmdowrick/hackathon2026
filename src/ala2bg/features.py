"""Permutation-invariant ion features: the input to the SRV progress coordinate.

WHY. Ions of one species are indistinguishable. Raw per-ion coordinates
(z1, z2, z3) change discontinuously when the trajectory's arbitrary atom
indices are relabelled -- measured at ~29x on KcsA -- and an SRV or MSM built
on them learns the labelling, not the physics. Every feature here is a SUM
over ions, so relabelling is exactly nothing.

    occ_s  = sum_a m(z_a, r_a) * w_s(z_a)      soft occupancy of basis site s
    n      = sum_a m(z_a, r_a)                 ions in the channel
    Q      = sum_a m(z_a, r_a) * delta(z_a)    displacement charge (delta.py)

m is a smooth channel membership (axial window x RADIAL GATE) and w_s a
partition of unity over evenly spaced basis centres, so sum_s occ_s = n.

WHY NOT SORTED POSITIONS. "The k innermost ions, sorted" is invariant too,
but it is discontinuous whenever an ion enters or leaves: a slot jumps from
its padding value to a real position. An SRV would read every entry event as
a teleport. Soft occupancies change continuously as an ion crosses the window
edge, which test_features_are_continuous_through_channel_entry checks.

WHY A RADIAL GATE, ALWAYS. Bug 2, three times over: without it, bulk ions at
the right z but nowhere near the pore were counted as channel ions and the
measured D came out 29x high. in_channel() is the ONE implementation; delta.py
uses it too.

WHY BASIS CENTRES AND NOT SITES. The binding sites are what the SRV and MSM
should discover. Evenly spaced centres, finer than the site spacing, are a
basis; nothing about the sites goes in.

ATOM INDEXING. Invariance is for the CV/state space only. Diffusivity must
still be estimated from atom-tracked trajectories (HANDOVER section 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class Channel:
    """The region in the pore frame where an ion counts as a channel ion.

    z_lo, z_hi : axial extent, A
    r_max      : radial gate, A from the pore axis
    width      : softness of every edge, A. 0 gives hard membership.
    """
    z_lo: float
    z_hi: float
    r_max: float
    width: float = 0.5

    def __post_init__(self):
        if not self.z_hi > self.z_lo:
            raise ValueError("z_hi must exceed z_lo")
        if not self.r_max > 0:
            raise ValueError("r_max must be positive: a channel without a "
                             "radial gate counts bulk ions (bug 2)")


def in_channel(z, r, ch: Channel, soft=True):
    """Membership of each ion in the channel, in [0, 1]. NaN (absent) -> 0."""
    z = np.asarray(z, float)
    r = np.asarray(r, float)
    present = np.isfinite(z) & np.isfinite(r)
    zz = np.where(present, z, 0.0)
    rr = np.where(present, r, np.inf)
    if not soft or ch.width <= 0:
        m = (zz >= ch.z_lo) & (zz <= ch.z_hi) & (rr <= ch.r_max)
        return np.where(present, m, False).astype(float)
    w = ch.width
    m = (_sigmoid((zz - ch.z_lo) / w) * _sigmoid((ch.z_hi - zz) / w)
         * _sigmoid((ch.r_max - np.where(np.isfinite(rr), rr, 1e9)) / w))
    return np.where(present, m, 0.0)


def to_pore_frame(xyz, origin, axis, box=None):
    """(z, r) of each ion in the pore frame.

    xyz  : (n_frames, n_ions, 3), A
    box  : (3,) or (n_frames, 3) orthorhombic box lengths. When given, each
           displacement from the origin is taken to its MINIMUM IMAGE first,
           so an ion that has crossed a periodic boundary is not reported a
           box length away from a pore it is sitting in (bug 7's class).
    """
    X = np.asarray(xyz, float)
    if X.ndim == 2:
        X = X[None]
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    d = X - np.asarray(origin, float)
    if box is not None:
        L = np.asarray(box, float)
        L = L[:, None, :] if L.ndim == 2 else L[None, None, :]
        d = d - L * np.round(d / L)
    z = d @ axis
    r = np.linalg.norm(d - z[..., None] * axis, axis=-1)
    return z, r


def basis_centres(ch: Channel, spacing=1.0):
    """Evenly spaced basis centres across the channel window."""
    n = max(int(np.floor((ch.z_hi - ch.z_lo) / spacing)) + 1, 2)
    return np.linspace(ch.z_lo, ch.z_hi, n)


def site_occupancy(z, r, ch: Channel, centres, sigma=None):
    """Soft occupancy of each basis centre, (n_frames, n_centres).

    Each channel ion is shared among the centres by a normalised Gaussian
    weight, so the occupancies of a frame sum to its soft ion count.
    """
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    r = np.asarray(r, float).reshape(z.shape)
    c = np.asarray(centres, float)
    if sigma is None:
        sigma = float(np.min(np.diff(c))) if len(c) > 1 else 1.0
    m = in_channel(z, r, ch)                              # (f, a)
    zz = np.where(np.isfinite(z), z, 0.0)
    g = -0.5 * ((zz[..., None] - c) / sigma) ** 2         # (f, a, s)
    g -= g.max(axis=-1, keepdims=True)
    w = np.exp(g)
    w /= w.sum(axis=-1, keepdims=True)
    return np.einsum("fa,fas->fs", m, w)


def invariant_features(z, r, ch: Channel, spacing=1.0, delta_fn=None):
    """The SRV input: basis occupancies, ion count, and optionally Q.

    Returns (X, names). Q is included when delta_fn (delta.delta_profile_fn)
    is given; it is the coordinate the membrane voltage couples to, so it is
    worth handing the SRV directly rather than hoping it is rediscovered.
    """
    c = basis_centres(ch, spacing)
    occ = site_occupancy(z, r, ch, c)
    cols = [occ, occ.sum(axis=1, keepdims=True)]
    names = [f"occ@{x:+.1f}A" for x in c] + ["n_channel"]
    if delta_fn is not None:
        z = np.asarray(z, float).reshape(occ.shape[0], -1)
        m = in_channel(z, np.asarray(r, float).reshape(z.shape), ch)
        d = delta_fn(np.where(np.isfinite(z), z, ch.z_lo))
        cols.append((m * d).sum(axis=1, keepdims=True))
        names.append("Q")
    return np.hstack(cols), names
