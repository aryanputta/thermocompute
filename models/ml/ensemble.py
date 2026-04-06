"""
Ensemble Models for Thermodynamic Property Prediction

Wraps Random Forest and Gradient Boosting regressors with:
- Standard feature scaling
- Cross-validated hyperparameter selection
- Feature importance reporting
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
from typing import Literal


@dataclass
class EnsemblePrediction:
    mean: np.ndarray
    model_name: str


class EnsembleThermo:
    """
    Tabular ensemble model for thermodynamic property prediction.

    Parameters
    ----------
    model : 'random_forest' or 'gradient_boosting'
    n_estimators : int
    max_depth : int or None
    random_state : int
    """

    def __init__(
        self,
        model: Literal["random_forest", "gradient_boosting"] = "random_forest",
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        self.model_name = model
        if model == "random_forest":
            self._model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model == "gradient_boosting":
            self._model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth or 4,
                random_state=random_state,
                learning_rate=0.05,
                subsample=0.8,
            )
        else:
            raise ValueError(f"Unknown model: {model}. Use 'random_forest' or 'gradient_boosting'")

        self._scaler = StandardScaler()
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "EnsembleThermo":
        """
        Fit the ensemble to training data.

        Parameters
        ----------
        X : (N, n_features) feature matrix
        y : (N,) target values
        feature_names : optional list of feature names for importance reporting
        """
        self._validate(X, y)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """Predict target property for input X."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return EnsemblePrediction(mean=preds, model_name=self.model_name)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """
        Run k-fold cross-validation and return MAE and RMSE metrics.

        Parameters
        ----------
        X : (N, n_features)
        y : (N,)
        cv : number of folds

        Returns
        -------
        dict with keys 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std'
        """
        self._validate(X, y)
        X_scaled = self._scaler.fit_transform(X)

        mae_scores = -cross_val_score(
            self._model, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        rmse_scores = np.sqrt(
            -cross_val_score(
                self._model, X_scaled, y, cv=cv, scoring="neg_mean_squared_error"
            )
        )
        return {
            "mae_mean": float(mae_scores.mean()),
            "mae_std": float(mae_scores.std()),
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_std": float(rmse_scores.std()),
        }

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances sorted descending."""
        self._check_fitted()
        imp = self._model.feature_importances_
        return dict(sorted(zip(self._feature_names, imp), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have equal length")


# Allow Optional import without re-importing typing
from typing import Optional
