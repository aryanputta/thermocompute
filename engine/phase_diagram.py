"""
Phase Diagram Generator

Constructs binary and ternary phase diagrams via:
- Gibbs energy surface sampling across composition space
- Convex hull construction to identify stable phases
- Common tangent construction for binary miscibility gaps
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from dataclasses import dataclass, field
from typing import Optional

from engine.gibbs_solver import GibbsSolver, Phase, Species, R


@dataclass
class BinaryDiagramResult:
    """Result of a binary phase diagram calculation."""
    x_grid: np.ndarray            # mole fractions of component B, shape (N,)
    G_mix: np.ndarray             # ΔG_mix (J/mol), shape (N,)
    stable_mask: np.ndarray       # True if point is on convex hull (stable), shape (N,)
    miscibility_gap: Optional[tuple[float, float]]  # (x_spinodal_L, x_spinodal_R) or None
    T: float


@dataclass
class TernaryDiagramResult:
    """Result of a ternary phase diagram calculation."""
    compositions: np.ndarray   # shape (N, 3) — barycentric coords
    G_mix: np.ndarray          # ΔG_mix (J/mol), shape (N,)
    hull_vertices: np.ndarray  # indices of stable (hull) compositions
    T: float


class PhaseDiagram:
    """
    Compute phase stability diagrams for binary and ternary systems.

    Parameters
    ----------
    phase : Phase
        The solution phase to analyze (liquid, solid solution, etc.)
    n_grid : int
        Number of composition grid points per axis
    """

    def __init__(self, phase: Phase, n_grid: int = 200):
        self.phase = phase
        self.n_grid = n_grid

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    def binary(self, T: float) -> BinaryDiagramResult:
        """
        Compute binary G_mix and identify miscibility gaps at temperature T.

        Assumes phase contains exactly 2 species.
        """
        if len(self.phase.species) != 2:
            raise ValueError(
                f"Binary diagram requires exactly 2 species; "
                f"phase '{self.phase.name}' has {len(self.phase.species)}"
            )

        x = np.linspace(0.0, 1.0, self.n_grid)
        G_mix = np.array([
            self.phase.gibbs_mixing(T, np.array([1 - xi, xi]))
            for xi in x
        ])

        # Convex hull in 2D: (x, G_mix)
        points = np.column_stack([x, G_mix])
        # Lower convex hull: include endpoints
        stable_mask = self._lower_convex_hull_mask(points)

        # Miscibility gap: region where hull departs from G_mix curve
        gap = self._find_miscibility_gap(x, G_mix, stable_mask)

        return BinaryDiagramResult(
            x_grid=x,
            G_mix=G_mix,
            stable_mask=stable_mask,
            miscibility_gap=gap,
            T=T,
        )

    def binary_vs_temperature(
        self, T_range: np.ndarray
    ) -> dict[float, BinaryDiagramResult]:
        """Compute binary diagram at each temperature in T_range."""
        return {T: self.binary(T) for T in T_range}

    # ------------------------------------------------------------------
    # Ternary
    # ------------------------------------------------------------------

    def ternary(self, T: float) -> TernaryDiagramResult:
        """
        Compute ternary G_mix surface and identify stable compositions.

        Assumes phase contains exactly 3 species.
        """
        if len(self.phase.species) != 3:
            raise ValueError(
                f"Ternary diagram requires exactly 3 species; "
                f"phase '{self.phase.name}' has {len(self.phase.species)}"
            )

        comps = self._ternary_grid()          # (N, 3)
        G_mix = np.array([
            self.phase.gibbs_mixing(T, c) for c in comps
        ])

        # 3D convex hull: (x1, x2, G_mix)
        # Stable compositions lie on the lower convex hull
        pts = np.column_stack([comps[:, 0], comps[:, 1], G_mix])
        try:
            hull = ConvexHull(pts)
            # Lower hull: facets with downward-pointing normal in G dimension
            lower_vertices = set()
            for simplex, eq in zip(hull.simplices, hull.equations):
                if eq[-2] < 0:   # normal z-component negative → lower face
                    lower_vertices.update(simplex)
            hull_vertices = np.array(sorted(lower_vertices))
        except Exception:
            hull_vertices = np.arange(len(comps))

        return TernaryDiagramResult(
            compositions=comps,
            G_mix=G_mix,
            hull_vertices=hull_vertices,
            T=T,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ternary_grid(self) -> np.ndarray:
        """Generate barycentric grid points for a ternary system."""
        n = self.n_grid
        pts = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                pts.append([i / n, j / n, k / n])
        return np.array(pts)

    @staticmethod
    def _lower_convex_hull_mask(points: np.ndarray) -> np.ndarray:
        """
        Return boolean mask: True if point is on the lower convex hull.
        points: (N, 2) array of (x, y).
        """
        n = len(points)
        if n < 3:
            return np.ones(n, dtype=bool)

        # Build lower hull via Graham scan
        sorted_idx = np.argsort(points[:, 0])
        hull_indices = set()

        lower: list[int] = []
        for idx in sorted_idx:
            while (
                len(lower) >= 2
                and _cross(points[lower[-2]], points[lower[-1]], points[idx]) <= 0
            ):
                lower.pop()
            lower.append(idx)

        hull_indices.update(lower)
        mask = np.zeros(n, dtype=bool)
        mask[list(hull_indices)] = True
        return mask

    @staticmethod
    def _find_miscibility_gap(
        x: np.ndarray,
        G_mix: np.ndarray,
        stable_mask: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        """
        Identify miscibility gap as region where G_mix lies above convex hull.
        Returns (x_left, x_right) of gap, or None if system is fully miscible.
        """
        # Unstable points: on curve but above hull
        unstable = ~stable_mask
        # Exclude endpoints
        unstable[[0, -1]] = False

        if not unstable.any():
            return None

        unstable_x = x[unstable]
        return (float(unstable_x.min()), float(unstable_x.max()))


def _cross(O: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """2D cross product of vectors OA and OB."""
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
