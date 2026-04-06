"""
Thermodynamic Data Pipeline

Transforms raw thermodynamic data (CSV / API responses) into
clean, feature-engineered arrays ready for model input.

Pipeline stages:
    raw_data → cleaned_data → feature_engineered → model_input

Features engineered:
    - Temperature-normalized terms (T/T_ref, ln(T))
    - Atomic features: radius, electronegativity, oxidation state
    - Composition-derived: mole fractions, x_i * x_j cross terms
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Atomic properties for selected rare-earth and common elements
# Sources: WebElements / NIST standard reference
ATOMIC_DATA: dict[str, dict] = {
    "La": {"Z": 57, "radius": 1.87, "electronegativity": 1.10, "oxidation_state": 3},
    "Ce": {"Z": 58, "radius": 1.82, "electronegativity": 1.12, "oxidation_state": 3},
    "Pr": {"Z": 59, "radius": 1.82, "electronegativity": 1.13, "oxidation_state": 3},
    "Nd": {"Z": 60, "radius": 1.81, "electronegativity": 1.14, "oxidation_state": 3},
    "Sm": {"Z": 62, "radius": 1.80, "electronegativity": 1.17, "oxidation_state": 3},
    "Eu": {"Z": 63, "radius": 1.99, "electronegativity": 1.20, "oxidation_state": 3},
    "Gd": {"Z": 64, "radius": 1.79, "electronegativity": 1.20, "oxidation_state": 3},
    "Dy": {"Z": 66, "radius": 1.75, "electronegativity": 1.22, "oxidation_state": 3},
    "Y":  {"Z": 39, "radius": 1.80, "electronegativity": 1.22, "oxidation_state": 3},
    "Fe": {"Z": 26, "radius": 1.26, "electronegativity": 1.83, "oxidation_state": 3},
    "O":  {"Z":  8, "radius": 0.66, "electronegativity": 3.44, "oxidation_state": -2},
    "Cl": {"Z": 17, "radius": 0.99, "electronegativity": 3.16, "oxidation_state": -1},
}

T_REF = 298.15  # K


@dataclass
class PipelineConfig:
    elements: list[str]                   # components in the system
    target_column: str = "G_mix"          # target property column name
    T_column: str = "T_K"
    test_fraction: float = 0.2
    random_seed: int = 42


@dataclass
class PipelineOutput:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)


class DataPipeline:
    """
    Full data pipeline from raw CSV to model-ready arrays.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, raw_path: Path) -> PipelineOutput:
        """
        Execute the full pipeline from a raw data CSV.

        Expected CSV columns:
            T_K  — temperature in Kelvin
            <element>_x  — mole fraction columns, e.g. La_x, Ce_x
            G_mix  — target (or other thermodynamic property)

        Returns
        -------
        PipelineOutput with train/test splits
        """
        df = self._load(raw_path)
        df = self._clean(df)
        X, y, feature_names = self._engineer_features(df)
        return self._split(X, y, feature_names)

    def run_from_dataframe(self, df: pd.DataFrame) -> PipelineOutput:
        """Run pipeline from an in-memory DataFrame."""
        df = self._clean(df)
        X, y, feature_names = self._engineer_features(df)
        return self._split(X, y, feature_names)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_csv(path)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        required = [cfg.T_column, cfg.target_column] + [
            f"{el}_x" for el in cfg.elements
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Drop rows with NaN in required columns
        df = df[required].dropna().copy()

        # Physical validity checks
        T = df[cfg.T_column].values
        if (T <= 0).any():
            raise ValueError("Temperature must be positive (K)")

        for el in cfg.elements:
            col = f"{el}_x"
            x = df[col].values
            if (x < 0).any() or (x > 1).any():
                raise ValueError(f"Mole fractions out of [0,1] for column {col}")

        # Verify mole fractions sum to ~1
        x_sum = df[[f"{el}_x" for el in cfg.elements]].sum(axis=1)
        if not np.allclose(x_sum.values, 1.0, atol=1e-4):
            raise ValueError(
                "Mole fractions do not sum to 1. Check raw data."
            )

        return df.reset_index(drop=True)

    def _engineer_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cfg = self.config
        features: dict[str, np.ndarray] = {}

        T = df[cfg.T_column].values
        features["T"] = T
        features["T_norm"] = T / T_REF
        features["ln_T"] = np.log(T)
        features["inv_T"] = 1.0 / T

        # Mole fractions
        x_dict: dict[str, np.ndarray] = {}
        for el in cfg.elements:
            x = df[f"{el}_x"].values
            features[f"x_{el}"] = x
            x_dict[el] = x

        # Cross terms x_i * x_j
        els = cfg.elements
        for i, el_i in enumerate(els):
            for j, el_j in enumerate(els):
                if j > i:
                    features[f"x_{el_i}_x_{el_j}"] = x_dict[el_i] * x_dict[el_j]

        # Composition-weighted atomic features
        for prop in ("radius", "electronegativity", "oxidation_state"):
            weighted = np.zeros(len(df))
            for el in cfg.elements:
                if el in ATOMIC_DATA:
                    weighted += x_dict[el] * ATOMIC_DATA[el][prop]
            features[f"avg_{prop}"] = weighted

        feature_names = list(features.keys())
        X = np.column_stack(list(features.values()))
        y = df[cfg.target_column].values.astype(float)
        return X, y, feature_names

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> PipelineOutput:
        rng = np.random.default_rng(self.config.random_seed)
        n = len(X)
        n_test = max(1, int(n * self.config.test_fraction))
        test_idx = rng.choice(n, size=n_test, replace=False)
        train_idx = np.setdiff1d(np.arange(n), test_idx)

        return PipelineOutput(
            X_train=X[train_idx],
            X_test=X[test_idx],
            y_train=y[train_idx],
            y_test=y[test_idx],
            feature_names=feature_names,
            metadata={
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(feature_names),
                "elements": self.config.elements,
            },
        )
