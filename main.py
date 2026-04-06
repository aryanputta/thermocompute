"""
ThermoCompute — Entry Point

Demonstrates the full pipeline:
1. Build a binary La-Ce ideal solution phase
2. Minimize Gibbs free energy at multiple temperatures
3. Generate a binary phase diagram
4. Fit a nonlinear regression model to synthetic ideal-solution data
5. Report prediction metrics vs ground truth

Run: python main.py
"""

from __future__ import annotations

import numpy as np
import sys
import os

# Ensure project root is on path when running directly
sys.path.insert(0, os.path.dirname(__file__))

from engine.gibbs_solver import GibbsSolver, Phase, Species, R
from engine.phase_diagram import PhaseDiagram
from models.regression.nonlinear import NonlinearThermodynamicRegressor


def make_lace_phase() -> Phase:
    """La-Ce binary ideal solution phase."""
    la = Species(
        name="La",
        formula="La",
        stoichiometry={"La": 1},
        g_standard_298=0.0,
    )
    ce = Species(
        name="Ce",
        formula="Ce",
        stoichiometry={"Ce": 1},
        g_standard_298=0.0,
    )
    return Phase(name="liquid", species=[la, ce])


def demo_gibbs_minimization():
    print("\n--- Gibbs Free Energy Minimization: La-Ce Binary ---")
    phase = make_lace_phase()
    solver = GibbsSolver([phase])
    feed = {"La": 0.6, "Ce": 0.4}

    for T in [500, 800, 1200, 1600]:
        result = solver.minimize(T=float(T), feed=feed)
        x = result.phase_compositions["liquid"]
        print(
            f"  T={T:5d} K | success={result.success} | "
            f"x_La={x[0]:.4f}  x_Ce={x[1]:.4f} | "
            f"G_total={result.G_total:.2f} J"
        )


def demo_phase_diagram():
    print("\n--- Binary Phase Diagram: La-Ce at 1000 K ---")
    phase = make_lace_phase()
    pd_gen = PhaseDiagram(phase, n_grid=100)
    result = pd_gen.binary(T=1000.0)

    idx_min = np.argmin(result.G_mix)
    print(f"  G_mix minimum: {result.G_mix[idx_min]:.2f} J/mol at x_Ce={result.x_grid[idx_min]:.3f}")
    print(f"  Miscibility gap: {result.miscibility_gap}")
    n_stable = result.stable_mask.sum()
    print(f"  Stable hull points: {n_stable} / {len(result.x_grid)}")


def demo_regression():
    print("\n--- Nonlinear Regression: La-Ce Ideal Solution ---")
    rng = np.random.default_rng(0)
    n = 100
    T_arr = rng.uniform(500, 1500, n)
    raw = rng.exponential(1.0, (n, 2))
    X_arr = raw / raw.sum(axis=1, keepdims=True)
    x_safe = np.clip(X_arr, 1e-15, 1.0)
    G_true = R * T_arr * np.sum(x_safe * np.log(x_safe), axis=1)

    # Train/test split
    split = 80
    model = NonlinearThermodynamicRegressor(n_components=2)
    model.fit(T_arr[:split], X_arr[:split], G_true[:split])
    G_pred = model.predict(T_arr[split:], X_arr[split:])

    mae = float(np.mean(np.abs(G_true[split:] - G_pred)))
    rmse = float(np.sqrt(np.mean((G_true[split:] - G_pred) ** 2)))
    print(f"  Test MAE:  {mae:.4f} J/mol")
    print(f"  Test RMSE: {rmse:.4f} J/mol")

    a01 = model.interaction_parameter(0, 1, T=1000.0)
    print(f"  Fitted L0(La,Ce) at 1000 K: {a01:.2f} J/mol  (ideal → ~0)")


if __name__ == "__main__":
    print("=" * 55)
    print("  ThermoCompute: Computational Thermodynamics Engine")
    print("=" * 55)

    demo_gibbs_minimization()
    demo_phase_diagram()
    demo_regression()

    print("\nDone. Run benchmarks: python benchmarks/benchmark.py")
    print("Run tests:            pytest tests/ -v")
