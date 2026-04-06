"""
Nonlinear Thermodynamic Regression

Fits a physics-informed parametric model to thermodynamic data:
    G_pred(T, x) = Σ_i x_i * G°_i(T) + RT * Σ_i x_i * ln(x_i)
                   + Σ_{i<j} x_i * x_j * (a_ij + b_ij * T)

Parameters a_ij, b_ij (Redlich-Kister L0 terms) are fit from data.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional

R = 8.314462  # J/(mol·K)


@dataclass
class RKParams:
    """Fitted Redlich-Kister L0 interaction parameters."""
    pair_indices: list[tuple[int, int]]  # (i, j) pairs
    a: np.ndarray   # enthalpy-like terms (J/mol)
    b: np.ndarray   # entropy-like terms (J/(mol·K))

    def interaction(self, i: int, j: int, T: float) -> float:
        """L0(T) = a_ij + b_ij * T  for pair (i,j)."""
        for k, (pi, pj) in enumerate(self.pair_indices):
            if (pi == i and pj == j) or (pi == j and pj == i):
                return self.a[k] + self.b[k] * T
        return 0.0


class NonlinearThermodynamicRegressor:
    """
    Fit Redlich-Kister L0 interaction parameters from (T, x, G_mix) data.

    Parameters
    ----------
    n_components : int
        Number of components in the system.
    g_standard : callable, optional
        g_standard(i, T) → G°_i(T) in J/mol.
        If None, G° is treated as 0 (pure G_mix fitting).
    """

    def __init__(
        self,
        n_components: int,
        g_standard: Optional[callable] = None,
    ):
        self.n_components = n_components
        self.g_standard = g_standard if g_standard is not None else lambda i, T: 0.0
        self.params_: Optional[RKParams] = None
        self._pair_indices = [
            (i, j) for i in range(n_components) for j in range(i + 1, n_components)
        ]
        self._n_params = 2 * len(self._pair_indices)  # a_ij, b_ij for each pair

    def fit(self, T_arr: np.ndarray, X_arr: np.ndarray, G_arr: np.ndarray):
        """
        Fit interaction parameters.

        Parameters
        ----------
        T_arr : (N,) temperatures in K
        X_arr : (N, n_components) mole fractions (rows sum to 1)
        G_arr : (N,) observed G_mix values (J/mol)
        """
        self._validate_inputs(T_arr, X_arr, G_arr)

        # Residual = G_obs - G_ideal - G_ref
        G_ideal = self._ideal_mixing(T_arr, X_arr)
        G_ref = self._reference(T_arr, X_arr)
        G_excess_obs = G_arr - G_ideal - G_ref

        def model(TX, *params):
            return self._excess_model(TX, params)

        # Pack (T, X) for curve_fit
        TX = np.column_stack([T_arr, X_arr])
        p0 = np.zeros(self._n_params)

        try:
            popt, pcov = curve_fit(
                model, TX.T, G_excess_obs, p0=p0, maxfev=10_000
            )
        except RuntimeError as e:
            raise RuntimeError(f"Regression did not converge: {e}") from e

        a = popt[: len(self._pair_indices)]
        b = popt[len(self._pair_indices):]
        self.params_ = RKParams(
            pair_indices=self._pair_indices,
            a=a,
            b=b,
        )
        return self

    def predict(self, T_arr: np.ndarray, X_arr: np.ndarray) -> np.ndarray:
        """
        Predict G_mix (J/mol) for given temperatures and compositions.

        Returns
        -------
        G_pred : (N,) array
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        self._validate_inputs(T_arr, X_arr)
        G_ideal = self._ideal_mixing(T_arr, X_arr)
        G_ref = self._reference(T_arr, X_arr)
        TX = np.column_stack([T_arr, X_arr])
        G_excess = self._excess_model(TX.T, tuple(np.concatenate([self.params_.a, self.params_.b])))
        return G_ideal + G_ref + G_excess

    def interaction_parameter(self, i: int, j: int, T: float) -> float:
        """Return fitted L0(i,j,T) in J/mol."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.params_.interaction(i, j, T)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ideal_mixing(self, T_arr: np.ndarray, X_arr: np.ndarray) -> np.ndarray:
        x_safe = np.clip(X_arr, 1e-15, 1.0)
        return R * T_arr * np.sum(x_safe * np.log(x_safe), axis=1)

    def _reference(self, T_arr: np.ndarray, X_arr: np.ndarray) -> np.ndarray:
        G_ref = np.zeros(len(T_arr))
        for i in range(self.n_components):
            G_ref += X_arr[:, i] * np.array([self.g_standard(i, T) for T in T_arr])
        return G_ref

    def _excess_model(self, TX_T: np.ndarray, params) -> np.ndarray:
        """
        TX_T: (n_features, N) — transpose of (T, x_0, x_1, ...) matrix.
        params: flat array [a_00, a_01, ..., b_00, b_01, ...]
        """
        T_arr = TX_T[0]
        X_arr = TX_T[1:].T   # (N, n_components)
        n_pairs = len(self._pair_indices)
        a = np.array(params[:n_pairs])
        b = np.array(params[n_pairs:])

        G_ex = np.zeros(len(T_arr))
        for k, (i, j) in enumerate(self._pair_indices):
            L0 = a[k] + b[k] * T_arr
            G_ex += X_arr[:, i] * X_arr[:, j] * L0

        return G_ex

    def _validate_inputs(self, T_arr, X_arr, G_arr=None):
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_components:
            raise ValueError(
                f"X_arr must be (N, {self.n_components}), got {X_arr.shape}"
            )
        if not np.allclose(X_arr.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1 along axis=1")
        if G_arr is not None and len(G_arr) != len(T_arr):
            raise ValueError("T_arr and G_arr must have the same length")
