"""Periodic coordinates with an ARBITRARY period, not just 2*pi.

WHY
---
cvs.py assumes angles on [-pi, pi), which is right for dihedrals and wrong for
everything else. An ion permeating a channel in a periodic simulation box has a
z-coordinate that is periodic with the BOX LENGTH, and the permeation cycle
(S4 -> S3 -> S2 -> S1 -> exit -> bulk -> re-enter) is a loop around that period.

Rescaling z to an angle by

    theta = 2*pi * (z - z0) / L

maps the problem onto the torus machinery already in this package, unchanged:
the same wrapping, the same sin/cos embedding, the same structural periodicity
of the learned H, and -- the point -- the same non-contractible cycle test.

WHY THE CYCLE TEST STOPS BEING A FORMALITY
------------------------------------------
On dihedrals, a winding cycle is a mathematical curiosity: the system does not
systematically circulate, so zero cycle affinity is expected and the test only
has teeth against an unconstrained baseline.

For ion permeation it is the observable. A net current IS a non-zero cycle flux
around the periodic z-loop. At zero applied voltage, detailed balance forces
the cycle affinity to vanish and there can be no steady current. Under an
applied field the affinity is non-zero and the current is its consequence. So
the Wegscheider residual becomes a physical measurement rather than a
numerical check, and the electrical work per cycle is what a bond-graph TF
element transduces.

units: this module is unit-agnostic. Feed it nanometres or angstroms, keep the
period in the same unit, and the resulting angles are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json

import numpy as np

TWO_PI = 2.0 * np.pi


def to_angle(z, period, origin=0.0):
    """Map a periodic coordinate onto [-pi, pi), seam at the window EDGES.

    theta = wrap(2 pi (z - origin)/period - pi), so

        z = origin            -> theta = -pi   (the seam)
        z = origin + period/2 -> theta = 0     (the middle)

    The -pi offset matters physically. Without it the seam falls in the MIDDLE
    of the window -- for an ion channel, right through the selectivity filter,
    which is precisely where the landscape must be continuous.
    """
    z = np.asarray(z, dtype=np.float64)
    th = TWO_PI * (z - origin) / period - np.pi
    return (th + np.pi) % TWO_PI - np.pi


def from_angle(theta, period, origin=0.0):
    """Inverse of to_angle, returning a value in [origin, origin + period).

    NOTE the modulo: (theta % 2pi), NOT ((theta + pi) % 2pi). The latter
    introduces a HALF-PERIOD SHIFT, so from_angle(to_angle(z)) came back
    displaced by period/2. Every value was still in range and the set of
    values was still correct -- only their ORDER changed -- so a round-trip
    test that compares sorted arrays passes happily while every coordinate is
    attached to the wrong position.

    That bug silently attributed the measured drift to positions half a window
    away, which left the learned landscape uncorrelated with the truth
    (corr -0.15 at the right magnitude) and no amount of retraining fixed it.
    test_arbitrary_period_round_trip now compares elementwise.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return origin + ((theta + np.pi) % TWO_PI) / TWO_PI * period


def wrap_period(d, period):
    """Shortest displacement under an arbitrary period.

    The same bug as on the torus, in different units: a naive difference
    reports a jump of ~L whenever an ion crosses the box boundary, which for a
    permeation study is EVERY productive event.
    """
    d = np.asarray(d, dtype=np.float64)
    return (d + 0.5 * period) % period - 0.5 * period


@dataclass(frozen=True)
class MixedCVSpec:
    """CVs with per-dimension periods, mixing angles and lengths.

    periods[k] is the period of dimension k in its own unit; use 2*pi for a
    dihedral. A non-periodic coordinate is NOT supported here on purpose --
    the whole package relies on periodicity for the structural guarantees, and
    silently treating an unbounded coordinate as periodic would wrap distant
    states onto each other. Bound such a coordinate first, or state the box.
    """

    names: tuple = ("z1", "z2")
    periods: tuple = (1.0, 1.0)
    origins: tuple = (0.0, 0.0)
    units: tuple = ("nm", "nm")
    n_harmonics: int = 2
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.names)
        for attr in ("periods", "origins", "units"):
            if len(getattr(self, attr)) != n:
                raise ValueError(f"{attr} must have {n} entries")
        if any(p <= 0 for p in self.periods):
            raise ValueError("periods must be positive")

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def n_features(self) -> int:
        return 2 * self.dim * self.n_harmonics

    def to_angles(self, X) -> np.ndarray:
        """(n, d) raw coordinates -> (n, d) angles on [-pi, pi)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.stack([to_angle(X[:, k], self.periods[k], self.origins[k])
                         for k in range(self.dim)], axis=1)

    def from_angles(self, TH) -> np.ndarray:
        TH = np.atleast_2d(np.asarray(TH, dtype=np.float64))
        return np.stack([from_angle(TH[:, k], self.periods[k], self.origins[k])
                         for k in range(self.dim)], axis=1)

    def scale_to_angle(self) -> np.ndarray:
        """d(theta)/d(z) per dimension.

        Needed to convert a diffusivity measured in angle units back to
        physical units: D_z = D_theta * (dz/dtheta)^2.
        """
        return TWO_PI / np.asarray(self.periods, dtype=np.float64)

    def D_angle_to_physical(self, D_angle):
        """Convert a diffusivity from angle^2/time to (unit^2)/time."""
        return np.asarray(D_angle) / self.scale_to_angle() ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("names", "periods", "origins", "units"):
            d[k] = list(getattr(self, k))
        return d

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def permeation_cycle_charge(n_ions_per_cycle: int, z_ion: float = 1.0):
    """Charge transferred per completed permeation cycle, in units of e.

    This is the modulus of the bond-graph TF element that couples the chemical
    cycle to the electrical domain: one loop around the periodic z-coordinate
    moves n_ions_per_cycle * z_ion elementary charges across the membrane, so
    the electrical work per cycle is that charge times the membrane potential.

    It is also what makes the cycle affinity measurable: under an applied
    potential V the affinity around the loop is n * z * F * V, which must be
    zero when V = 0 if the model is thermodynamically consistent.
    """
    return float(n_ions_per_cycle * z_ion)
