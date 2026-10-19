"""Collective variables on the (phi, psi) torus.

THE SINGLE IMPLEMENTATION. Every stage imports wrapping and embedding from here.
The CVSpec hash is written into every artifact; stage 60 refuses to run on a mismatch.

The torus is the whole point: phi, psi in [-pi, pi) with both ends identified.
Getting the wrapping wrong silently corrupts every transition that crosses the
boundary -- which is most of the interesting ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(a: np.ndarray) -> np.ndarray:
    """Map angles into [-pi, pi). The one true wrapping function.

    >>> float(wrap(np.array([np.pi + 0.1]))[0]) < 0
    True
    """
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


def wrapped_delta(x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Shortest displacement x1 - x0 on the torus, componentwise.

    A naive x1 - x0 reports a jump of ~2pi whenever a trajectory crosses the
    periodic boundary. That single bug would wreck both the diffusion estimate
    and the PHNN drift-matching loss.
    """
    return wrap(np.asarray(x1) - np.asarray(x0))


def fourier_features(x: np.ndarray, n_harmonics: int = 2) -> np.ndarray:
    """Embed angles on the torus as sin/cos harmonics.

    x : (..., d) angles in radians.
    returns (..., 2 * d * n_harmonics)

    This is what makes periodicity STRUCTURAL for the neural network: a function
    of these features is automatically single-valued on the torus, so it cannot
    develop a spurious winding and is continuous across +-pi. A plain MLP on raw
    angles is discontinuous there.
    """
    x = np.asarray(x, dtype=np.float64)
    out = []
    for k in range(1, n_harmonics + 1):
        out.append(np.sin(k * x))
        out.append(np.cos(k * x))
    return np.concatenate(out, axis=-1)


@dataclass(frozen=True)
class CVSpec:
    """Frozen description of the collective variables.

    `.hash()` is the contract: it covers names, units, the periodic convention
    and the harmonic count, so any downstream artifact can prove it was built
    against the same CV definition.
    """

    names: tuple = ("phi", "psi")
    units: str = "radian"
    domain: tuple = (-np.pi, np.pi)
    periodic: tuple = (True, True)
    n_harmonics: int = 2
    source: str = "mdtraj.compute_phi/compute_psi on ACE-ALA-NME"
    extra: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def n_features(self) -> int:
        return 2 * self.dim * self.n_harmonics

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = [float(v) for v in self.domain]
        d["names"] = list(self.names)
        d["periodic"] = [bool(v) for v in self.periodic]
        return d

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def embed(self, x: np.ndarray) -> np.ndarray:
        return fourier_features(x, self.n_harmonics)


DEFAULT_CV = CVSpec()
