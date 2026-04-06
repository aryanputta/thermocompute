"""
Gibbs Free Energy Minimization Solver

Minimizes G_total = Σ(n_i * μ_i) over all species and phases
subject to mass balance and non-negativity constraints.

Supports ideal and non-ideal (Margules) solution models.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from dataclasses import dataclass, field
from typing import Optional


R = 8.314462  # J/(mol·K)


@dataclass
class Species:
    """A thermodynamic species with standard-state properties."""
    name: str
    formula: str
    stoichiometry: dict[str, float]  # element -> count
    # Shomate coefficients [A, B, C, D, E, F, H_ref] for G°(T) (NIST form)
    # Valid over a temperature range
    shomate: Optional[np.ndarray] = None
    # Fallback: constant G° at 298 K (J/mol)
    g_standard_298: float = 0.0

    def g_standard(self, T: float) -> float:
        """
        Standard Gibbs free energy at temperature T (J/mol).
        Uses Shomate equation if coefficients are provided:
            H°(T) - H°(298) = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t + F - H
            S°(T) = A*ln(t) + B*t + C*t²/2 + D*t³/3 - E/(2t²) + G
            G°(T) = H°(T) - T*S°(T)
        where t = T/1000.
        """
        if self.shomate is None:
            # Approximate G°(T) ≈ G°(298) + (T - 298) * ΔS term (crude)
            return self.g_standard_298

        A, B, C, D, E, F, G_coef, H_ref = self.shomate
        t = T / 1000.0
        h = A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F - H_ref
        s = A * np.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G_coef
        # h in kJ/mol → convert to J/mol
        return (h * 1000.0) - T * s


@dataclass
class Phase:
    """A thermodynamic phase (e.g., liquid, solid, gas) containing species."""
    name: str
    species: list[Species]
    # Margules binary interaction parameters W[i][j] (J/mol), default ideal
    interaction: Optional[np.ndarray] = None

    def chemical_potential(self, T: float, x: np.ndarray) -> np.ndarray:
        """
        Chemical potential vector μ_i (J/mol) for all species in this phase.
        μ_i = G°_i(T) + RT·ln(x_i) + RT·ln(γ_i)   [ideal: γ_i = 1]
        """
        n = len(self.species)
        assert x.shape == (n,), f"Expected {n} mole fractions, got {x.shape}"

        # Clip to avoid log(0); physical constraint enforced via bounds
        x_safe = np.clip(x, 1e-15, 1.0)

        mu = np.array([sp.g_standard(T) for sp in self.species])
        mu += R * T * np.log(x_safe)

        if self.interaction is not None:
            # Margules activity coefficients (symmetric)
            # ln(γ_i) = Σ_{j≠i} W_ij * x_j² / (RT)  (one-suffix Margules)
            W = self.interaction
            for i in range(n):
                excess = 0.0
                for j in range(n):
                    if i != j:
                        excess += W[i, j] * x_safe[j] ** 2
                mu[i] += excess

        return mu

    def gibbs_mixing(self, T: float, x: np.ndarray) -> float:
        """
        Molar Gibbs energy of mixing ΔG_mix (J/mol).
        ΔG_mix = RT·Σ(x_i·ln(x_i)) + G_excess
        """
        x_safe = np.clip(x, 1e-15, 1.0)
        ideal = R * T * np.sum(x_safe * np.log(x_safe))

        excess = 0.0
        if self.interaction is not None:
            W = self.interaction
            n = len(x)
            for i in range(n):
                for j in range(i + 1, n):
                    excess += W[i, j] * x_safe[i] * x_safe[j]

        return ideal + excess


@dataclass
class EquilibriumResult:
    """Output of a Gibbs minimization run."""
    success: bool
    G_total: float                         # J
    phase_amounts: dict[str, float]        # phase_name -> total moles
    phase_compositions: dict[str, np.ndarray]  # phase_name -> mole fractions
    message: str = ""
    n_iterations: int = 0


class GibbsSolver:
    """
    Multi-phase Gibbs free energy minimizer.

    Minimizes G_total = Σ_α Σ_i  n_i^α · μ_i^α(T, x^α)
    subject to:
        mass balance:  Σ_α n_i^α = N_i   for each element
        non-negativity: n_i^α ≥ 0

    Parameters
    ----------
    phases : list of Phase
    method : 'SLSQP' or 'L-BFGS-B'
    tol : solver tolerance
    """

    def __init__(
        self,
        phases: list[Phase],
        method: str = "SLSQP",
        tol: float = 1e-10,
    ):
        self.phases = phases
        self.method = method
        self.tol = tol
        self._all_species = [sp for ph in phases for sp in ph.species]
        self._all_elements = sorted(
            {el for sp in self._all_species for el in sp.stoichiometry}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def minimize(
        self,
        T: float,
        feed: dict[str, float],
    ) -> EquilibriumResult:
        """
        Find equilibrium at temperature T given feed composition.

        Parameters
        ----------
        T : temperature in Kelvin
        feed : {element: total_moles}

        Returns
        -------
        EquilibriumResult
        """
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")

        n_vars, n0, bounds = self._build_variables(feed)
        A_eq, b_eq = self._build_mass_balance(feed)

        def objective(n_flat):
            return self._total_gibbs(n_flat, T)

        def gradient(n_flat):
            return self._total_gibbs_grad(n_flat, T)

        constraints = [
            {"type": "eq", "fun": lambda n: A_eq @ n - b_eq}
        ]

        result = minimize(
            objective,
            n0,
            jac=gradient,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            options={"maxiter": 2000, "ftol": self.tol},
        )

        return self._parse_result(result, T)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_variables(self, feed: dict[str, float]):
        """Build initial guess and bounds for molar amounts n_i^α."""
        n_total = sum(feed.values())
        n_species = [len(ph.species) for ph in self.phases]
        n_vars = sum(n_species)

        # Distribute feed equally across phases and species as initial guess
        n0 = np.full(n_vars, n_total / max(n_vars, 1) * 0.1)
        bounds = Bounds(lb=0.0, ub=np.inf)
        return n_vars, n0, bounds

    def _build_mass_balance(self, feed: dict[str, float]):
        """
        Build element mass balance matrix A and rhs b.
        Σ_α Σ_i  ν_{e,i} · n_i^α = N_e   for each element e
        """
        n_el = len(self._all_elements)
        n_vars = sum(len(ph.species) for ph in self.phases)

        A = np.zeros((n_el, n_vars))
        col = 0
        for ph in self.phases:
            for sp in ph.species:
                for e_idx, el in enumerate(self._all_elements):
                    A[e_idx, col] = sp.stoichiometry.get(el, 0.0)
                col += 1

        b = np.array([feed.get(el, 0.0) for el in self._all_elements])
        return A, b

    def _total_gibbs(self, n_flat: np.ndarray, T: float) -> float:
        """Compute total Gibbs free energy from flattened mole vector."""
        G = 0.0
        col = 0
        for ph in self.phases:
            n_ph = n_flat[col: col + len(ph.species)]
            n_total_ph = n_ph.sum()
            if n_total_ph < 1e-20:
                col += len(ph.species)
                continue
            x = n_ph / n_total_ph
            mu = ph.chemical_potential(T, x)
            G += np.dot(n_ph, mu)
            col += len(ph.species)
        return G

    def _total_gibbs_grad(self, n_flat: np.ndarray, T: float) -> np.ndarray:
        """Analytical gradient dG/dn_i^α = μ_i^α."""
        grad = np.zeros_like(n_flat)
        col = 0
        for ph in self.phases:
            n_ph = n_flat[col: col + len(ph.species)]
            n_total_ph = n_ph.sum()
            if n_total_ph < 1e-20:
                col += len(ph.species)
                continue
            x = n_ph / n_total_ph
            mu = ph.chemical_potential(T, x)
            grad[col: col + len(ph.species)] = mu
            col += len(ph.species)
        return grad

    def _parse_result(self, result, T: float) -> EquilibriumResult:
        col = 0
        phase_amounts = {}
        phase_compositions = {}
        for ph in self.phases:
            n_ph = result.x[col: col + len(ph.species)]
            n_total = n_ph.sum()
            phase_amounts[ph.name] = float(n_total)
            if n_total > 1e-15:
                phase_compositions[ph.name] = n_ph / n_total
            else:
                phase_compositions[ph.name] = np.zeros(len(ph.species))
            col += len(ph.species)

        return EquilibriumResult(
            success=result.success,
            G_total=float(result.fun),
            phase_amounts=phase_amounts,
            phase_compositions=phase_compositions,
            message=result.message,
            n_iterations=result.nit,
        )
