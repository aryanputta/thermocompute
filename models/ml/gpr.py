"""
Gaussian Process Regression for Thermodynamic Properties

Provides uncertainty-aware predictions of:
- Gibbs free energy surfaces
- Phase boundaries
- Solubility limits

Uses a Matérn 5/2 kernel with a white noise term.
Scales features before fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPRPrediction:
    mean: np.ndarray
    std: np.ndarray  # 1-sigma uncertainty


class GaussianProcessThermo:
    """
    GPR model for thermodynamic property prediction.

    Input features X: [T, P, x_1, x_2, ..., x_n, Z_1, ..., atomic_features]
    Target y: any scalar thermodynamic property (G, H, S, solubility, ...)

    Parameters
    ----------
    length_scale_bounds : tuple
        Bounds for kernel length scale optimization.
    n_restarts : int
        Number of optimizer restarts to avoid local minima.
    normalize_y : bool
        Whether to normalize the target variable.
    """

    def __init__(
        self,
        length_scale_bounds: tuple[float, float] = (1e-2, 1e3),
        n_restarts: int = 5,
        normalize_y: bool = True,
    ):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=length_scale_bounds, nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            alpha=0.0,  # noise is in kernel
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessThermo":
        """
        Fit GPR to training data.

        Parameters
        ----------
        X : (N, n_features) feature matrix
        y : (N,) target values
        """
        self._validate(X, y)
        X_scaled = self._scaler.fit_transform(X)
        self._gpr.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> GPRPrediction:
        """
        Predict with uncertainty estimates.

        Returns
        -------
        GPRPrediction with mean and std arrays of shape (N,)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        mean, std = self._gpr.predict(X_scaled, return_std=True)
        return GPRPrediction(mean=mean, std=std)

    def log_marginal_likelihood(self) -> float:
        """Return the log marginal likelihood of the fitted kernel."""
        self._check_fitted()
        return float(self._gpr.log_marginal_likelihood_value_)

    @property
    def kernel_params(self) -> dict:
        """Return fitted kernel parameters."""
        self._check_fitted()
        return dict(self._gpr.kernel_.get_params())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        if len(X) < 2:
            raise ValueError("Need at least 2 training samples for GPR")
