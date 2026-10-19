"""The single periodic grid on the (phi, psi) torus.

Contract B: defined once, used by the FES, the diffusion field, the PHNN
evaluation and the bond graph. Never redefined downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class TorusGrid:
    """Uniform periodic grid on [-pi, pi)^2.

    Cell centres, uniform cell area, and neighbour lists that WRAP in both
    directions. The wrapping is what makes the non-contractible cycles exist,
    which is what makes the Wegscheider test interesting.
    """

    n_phi: int = 60
    n_psi: int = 60

    @property
    def shape(self) -> tuple:
        return (self.n_phi, self.n_psi)

    @property
    def n_bins(self) -> int:
        return self.n_phi * self.n_psi

    @property
    def h(self) -> np.ndarray:
        """Spacing in each dimension."""
        return np.array([TWO_PI / self.n_phi, TWO_PI / self.n_psi])

    @property
    def cell_area(self) -> float:
        return float(self.h[0] * self.h[1])

    def edges(self):
        return (
            np.linspace(-np.pi, np.pi, self.n_phi + 1),
            np.linspace(-np.pi, np.pi, self.n_psi + 1),
        )

    def centers(self):
        ephi, epsi = self.edges()
        return 0.5 * (ephi[:-1] + ephi[1:]), 0.5 * (epsi[:-1] + epsi[1:])

    def center_grid(self) -> np.ndarray:
        """(n_bins, 2) array of cell-centre coordinates, row-major, phi slowest."""
        cphi, cpsi = self.centers()
        P, S = np.meshgrid(cphi, cpsi, indexing="ij")
        return np.stack([P.ravel(), S.ravel()], axis=1)

    def index_of(self, x: np.ndarray) -> np.ndarray:
        """Bin index for coordinates x (..., 2). Wraps by construction."""
        x = np.asarray(x)
        i = np.floor((x[..., 0] + np.pi) / self.h[0]).astype(int) % self.n_phi
        j = np.floor((x[..., 1] + np.pi) / self.h[1]).astype(int) % self.n_psi
        return self.ravel(i, j)

    def ravel(self, i, j):
        return np.asarray(i) * self.n_psi + np.asarray(j)

    def unravel(self, k):
        k = np.asarray(k)
        return k // self.n_psi, k % self.n_psi

    def neighbours(self):
        """Yield (i_flat, j_flat, axis) for each undirected edge, exactly once.

        Edges wrap: bin (0, j) neighbours (n_phi - 1, j). On a torus with
        n >= 3 in each direction every node has exactly 4 distinct neighbours
        and the edge count is 2 * n_bins.
        """
        for i in range(self.n_phi):
            for j in range(self.n_psi):
                a = self.ravel(i, j)
                # +phi
                b = self.ravel((i + 1) % self.n_phi, j)
                yield a, b, 0
                # +psi
                c = self.ravel(i, (j + 1) % self.n_psi)
                yield a, c, 1


DEFAULT_GRID = TorusGrid()
