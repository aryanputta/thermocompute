"""
Thermodynamic Data Loaders

Connects to external data sources:
    - NIST Chemistry WebBook (Shomate equation coefficients via HTTP)
    - Materials Project API (formation energies, stability data)

Requires:
    - mp-api package for Materials Project
    - MP_API_KEY environment variable for Materials Project access
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


NIST_BASE_URL = "https://webbook.nist.gov/cgi/cbook.cgi"


@dataclass
class ShomateCoefficients:
    """NIST Shomate equation coefficients for a species."""
    formula: str
    T_low: float    # K
    T_high: float   # K
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H_ref: float    # H°(298.15) kJ/mol

    def to_array(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H_ref])


class NISTLoader:
    """
    Retrieve Shomate coefficients from the NIST WebBook.

    Note: NIST WebBook does not have a formal REST API. This loader
    parses structured data that is available for specific compounds.
    For automated workflows, download NIST data files manually and
    use load_from_file().
    """

    @staticmethod
    def load_from_file(path: str) -> pd.DataFrame:
        """
        Load thermodynamic data from a local CSV exported from NIST.

        Expected columns: formula, T_low, T_high, A, B, C, D, E, F, G, H_ref

        Parameters
        ----------
        path : str
            Path to the CSV file.
        """
        df = pd.read_csv(path)
        required = ["formula", "T_low", "T_high", "A", "B", "C", "D", "E", "F", "G", "H_ref"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"NIST CSV missing columns: {missing}")
        return df

    @staticmethod
    def shomate_to_species_kwargs(row: pd.Series) -> dict:
        """Convert a NIST data row to kwargs for Species(shomate=...)."""
        coeffs = np.array([
            row["A"], row["B"], row["C"], row["D"],
            row["E"], row["F"], row["G"], row["H_ref"],
        ])
        return {
            "name": row["formula"],
            "formula": row["formula"],
            "shomate": coeffs,
        }


class MaterialsProjectLoader:
    """
    Retrieve formation energy data from the Materials Project API.

    Requires mp-api package:
        pip install mp-api

    Set MP_API_KEY environment variable before use.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Materials Project API key not found. "
                "Set the MP_API_KEY environment variable."
            )

    def fetch_formation_energies(self, formula: str) -> pd.DataFrame:
        """
        Fetch formation energies for all entries matching formula.

        Parameters
        ----------
        formula : str
            Chemical formula (e.g., 'La2O3', 'CeO2').

        Returns
        -------
        DataFrame with columns: mp_id, formula, e_above_hull, formation_energy_per_atom
        """
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError(
                "mp-api package not installed. Run: pip install mp-api"
            )

        with MPRester(self.api_key) as mpr:
            entries = mpr.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "e_above_hull", "formation_energy_per_atom"],
            )

        records = [
            {
                "mp_id": e.material_id,
                "formula": e.formula_pretty,
                "e_above_hull": e.e_above_hull,
                "formation_energy_per_atom": e.formation_energy_per_atom,
            }
            for e in entries
        ]
        return pd.DataFrame(records)

    def fetch_phase_diagram(self, elements: list[str]) -> pd.DataFrame:
        """
        Fetch all stable entries for a given set of elements.

        Parameters
        ----------
        elements : list of element symbols, e.g. ['La', 'Ce', 'O']

        Returns
        -------
        DataFrame with phase stability data
        """
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError("mp-api package not installed. Run: pip install mp-api")

        with MPRester(self.api_key) as mpr:
            entries = mpr.summary.search(
                elements=elements,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "e_above_hull",
                    "formation_energy_per_atom",
                    "is_stable",
                ],
            )

        records = [
            {
                "mp_id": e.material_id,
                "formula": e.formula_pretty,
                "e_above_hull": e.e_above_hull,
                "formation_energy_per_atom": e.formation_energy_per_atom,
                "is_stable": e.is_stable,
            }
            for e in entries
        ]
        return pd.DataFrame(records)
