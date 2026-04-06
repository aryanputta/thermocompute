"""
Benchmark Suite — ThermoCompute

Measures:
1. Prediction accuracy vs known ideal-solution values
   - MAE, RMSE, % deviation for nonlinear regression, GPR, and ensemble
2. Error reduction over iterative learning loop
3. Compute time vs accuracy tradeoff across model complexity
4. Comparison against naive baselines (mean predictor, linear interpolation)

Outputs results to benchmarks/results.csv.
"""

from __future__ import annotations

import time
import csv
import numpy as np
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.regression.nonlinear import NonlinearThermodynamicRegressor
from models.ml.gpr import GaussianProcessThermo
from models.ml.ensemble import EnsembleThermo

R = 8.314462  # J/(mol·K)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def ideal_gmix(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Ground truth: ideal solution G_mix."""
    x_safe = np.clip(X, 1e-15, 1.0)
    return R * T * np.sum(x_safe * np.log(x_safe), axis=1)


def generate_data(n: int, n_comp: int = 2, noise_std: float = 0.0, seed: int = 0):
    """
    Generate (T, X, G) dataset for an ideal n-component solution.

    Parameters
    ----------
    n : number of data points
    n_comp : number of components
    noise_std : additive Gaussian noise on G (J/mol), 0 for exact data
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    T = rng.uniform(500, 1500, n)
    raw = rng.exponential(1.0, (n, n_comp))
    X = raw / raw.sum(axis=1, keepdims=True)
    G = ideal_gmix(T, X)
    if noise_std > 0:
        G += rng.normal(0, noise_std, n)
    features = np.column_stack([T.reshape(-1, 1), X])
    return features, G, T, X


def build_train_test(
    n_train: int, n_test: int = 200, n_comp: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_train, G_train, T_train, X_train = generate_data(n_train, n_comp, seed=0)
    feat_test, G_test, _, _ = generate_data(n_test, n_comp, seed=1)
    return feat_train, G_train, feat_test, G_test, T_train, X_train


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pct_deviation(y_true, y_pred) -> float:
    """Mean absolute % deviation, ignoring near-zero points."""
    mask = np.abs(y_true) > 10.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------


def predict_mean_baseline(y_train, y_test) -> np.ndarray:
    """Always predict the training mean."""
    return np.full(len(y_test), y_train.mean())


def predict_linear_interp(T_train, G_train, T_test) -> np.ndarray:
    """1D linear interpolation on temperature (marginalizes composition)."""
    sort_idx = np.argsort(T_train)
    return np.interp(T_test, T_train[sort_idx], G_train[sort_idx])


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> dict:
    return {
        "model": name,
        "mae_J_mol": round(mae(y_true, y_pred), 4),
        "rmse_J_mol": round(rmse(y_true, y_pred), 4),
        "pct_deviation": round(pct_deviation(y_true, y_pred), 4),
        "time_s": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Benchmark 1: accuracy vs baselines
# ---------------------------------------------------------------------------


def bench_accuracy(n_train: int = 200, n_comp: int = 2) -> list[dict]:
    feat_train, G_train, feat_test, G_test, T_train, X_train = build_train_test(
        n_train, n_comp=n_comp
    )
    T_test = feat_test[:, 0]
    X_test = feat_test[:, 1:]

    results = []

    # Baseline 1: mean predictor
    t0 = time.perf_counter()
    pred_mean = predict_mean_baseline(G_train, G_test)
    results.append(evaluate_model("mean_baseline", G_test, pred_mean, time.perf_counter() - t0))

    # Baseline 2: linear interpolation
    t0 = time.perf_counter()
    pred_interp = predict_linear_interp(T_train, G_train, T_test)
    results.append(evaluate_model("linear_interp", G_test, pred_interp, time.perf_counter() - t0))

    # Model 1: nonlinear regression
    try:
        t0 = time.perf_counter()
        reg = NonlinearThermodynamicRegressor(n_components=n_comp)
        reg.fit(T_train, X_train, G_train)
        pred_reg = reg.predict(T_test, X_test)
        results.append(evaluate_model("nonlinear_regression", G_test, pred_reg, time.perf_counter() - t0))
    except Exception as e:
        results.append({"model": "nonlinear_regression", "error": str(e)})

    # Model 2: GPR (small n_train for speed)
    gpr_n = min(n_train, 100)
    try:
        t0 = time.perf_counter()
        gpr = GaussianProcessThermo(n_restarts=2)
        gpr.fit(feat_train[:gpr_n], G_train[:gpr_n])
        pred_gpr = gpr.predict(feat_test).mean
        results.append(evaluate_model("gpr", G_test, pred_gpr, time.perf_counter() - t0))
    except Exception as e:
        results.append({"model": "gpr", "error": str(e)})

    # Model 3: Random Forest
    try:
        t0 = time.perf_counter()
        rf = EnsembleThermo(model="random_forest", n_estimators=100)
        rf.fit(feat_train, G_train)
        pred_rf = rf.predict(feat_test).mean
        results.append(evaluate_model("random_forest", G_test, pred_rf, time.perf_counter() - t0))
    except Exception as e:
        results.append({"model": "random_forest", "error": str(e)})

    # Model 4: Gradient Boosting
    try:
        t0 = time.perf_counter()
        gb = EnsembleThermo(model="gradient_boosting", n_estimators=100)
        gb.fit(feat_train, G_train)
        pred_gb = gb.predict(feat_test).mean
        results.append(evaluate_model("gradient_boosting", G_test, pred_gb, time.perf_counter() - t0))
    except Exception as e:
        results.append({"model": "gradient_boosting", "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: error reduction over iterative learning loop
# ---------------------------------------------------------------------------


def bench_iterative_learning(n_comp: int = 2, max_points: int = 160) -> list[dict]:
    """
    Simulate adding 10 new experimental points per iteration.
    Track how RMSE decreases as more data is added.
    """
    feat_test, G_test, _, _ = generate_data(200, n_comp, seed=1)
    results = []

    for n_train in range(10, max_points + 1, 10):
        feat_train, G_train, _, _ = generate_data(n_train, n_comp, seed=0)
        try:
            t0 = time.perf_counter()
            rf = EnsembleThermo(model="random_forest", n_estimators=100)
            rf.fit(feat_train, G_train)
            pred = rf.predict(feat_test).mean
            elapsed = time.perf_counter() - t0
            results.append({
                "model": "iterative_random_forest",
                "n_train": n_train,
                "rmse_J_mol": round(rmse(G_test, pred), 4),
                "mae_J_mol": round(mae(G_test, pred), 4),
                "time_s": round(elapsed, 4),
            })
        except Exception as e:
            results.append({"model": "iterative_random_forest", "n_train": n_train, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: multi-component scaling
# ---------------------------------------------------------------------------


def bench_scaling() -> list[dict]:
    """Test accuracy and speed for 2, 3, 4 component systems."""
    results = []
    for n_comp in [2, 3, 4]:
        feat_train, G_train, feat_test, G_test, _, _ = build_train_test(
            n_train=200, n_comp=n_comp
        )
        try:
            t0 = time.perf_counter()
            rf = EnsembleThermo(model="random_forest", n_estimators=100)
            rf.fit(feat_train, G_train)
            pred = rf.predict(feat_test).mean
            elapsed = time.perf_counter() - t0
            results.append({
                "model": f"rf_{n_comp}comp",
                "n_components": n_comp,
                "rmse_J_mol": round(rmse(G_test, pred), 4),
                "mae_J_mol": round(mae(G_test, pred), 4),
                "time_s": round(elapsed, 4),
            })
        except Exception as e:
            results.append({"model": f"rf_{n_comp}comp", "n_components": n_comp, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_results(all_results: list[dict]):
    if not all_results:
        return
    fieldnames = sorted({k for r in all_results for k in r.keys()})
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"Results written to {RESULTS_PATH}")


def print_table(results: list[dict], title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  {r.get('model', '?')}: ERROR — {r['error']}")
        else:
            parts = [f"{k}={v}" for k, v in r.items() if k != "model"]
            print(f"  {r.get('model', '?')}: {', '.join(parts)}")


if __name__ == "__main__":
    print("Running ThermoCompute benchmarks...")

    results_accuracy = bench_accuracy(n_train=200, n_comp=2)
    print_table(results_accuracy, "Accuracy vs Baselines (binary, n_train=200)")

    results_iter = bench_iterative_learning(n_comp=2)
    print_table(results_iter, "Iterative Learning Loop (binary)")

    results_scale = bench_scaling()
    print_table(results_scale, "Multi-Component Scaling")

    all_results = results_accuracy + results_iter + results_scale
    write_results(all_results)
