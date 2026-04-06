"""
Physical Constraint Tests

Validates that solver and models satisfy hard physical laws:
- Mass conservation: element moles in == element moles out
- Non-negativity: no negative mole amounts
- Energy minimization: G_total ≤ G_initial
- Mole fractions sum to 1
- Phase diagram: G_mix ≤ 0 for ideal mixing at any composition
- GPR uncertainty: std > 0 for unseen points
- Data pipeline: output shapes and ranges are valid
"""

import pytest
import numpy as np
import pandas as pd

from engine.gibbs_solver import GibbsSolver, Phase, Species
from engine.phase_diagram import PhaseDiagram
from models.regression.nonlinear import NonlinearThermodynamicRegressor
from models.ml.gpr import GaussianProcessThermo
from models.ml.ensemble import EnsembleThermo
from data.pipeline import DataPipeline, PipelineConfig

R = 8.314462


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ideal_binary_phase() -> Phase:
    return Phase(
        name="liquid",
        species=[
            Species("La", "La", {"La": 1}, g_standard_298=0.0),
            Species("Ce", "Ce", {"Ce": 1}, g_standard_298=0.0),
        ],
    )


def ideal_ternary_phase() -> Phase:
    return Phase(
        name="liquid",
        species=[
            Species("La", "La", {"La": 1}, g_standard_298=0.0),
            Species("Ce", "Ce", {"Ce": 1}, g_standard_298=0.0),
            Species("Nd", "Nd", {"Nd": 1}, g_standard_298=0.0),
        ],
    )


def make_synthetic_data(n: int = 60, n_comp: int = 2, seed: int = 0) -> tuple:
    """
    Generate synthetic (T, X, G_mix) data for a simple ideal solution.
    T in [500, 1500] K, compositions random on simplex.
    """
    rng = np.random.default_rng(seed)
    T_arr = rng.uniform(500, 1500, n)
    # Random compositions on simplex
    raw = rng.exponential(1.0, (n, n_comp))
    X_arr = raw / raw.sum(axis=1, keepdims=True)
    # Ideal G_mix
    x_safe = np.clip(X_arr, 1e-15, 1.0)
    G_arr = R * T_arr * np.sum(x_safe * np.log(x_safe), axis=1)
    return T_arr, X_arr, G_arr


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------


class TestMassConservation:
    @pytest.mark.parametrize("feed", [
        {"La": 1.0, "Ce": 1.0},
        {"La": 0.3, "Ce": 0.7},
        {"La": 0.01, "Ce": 0.99},
        {"La": 2.5, "Ce": 0.5},
    ])
    def test_binary_element_conservation(self, feed):
        solver = GibbsSolver([ideal_binary_phase()])
        result = solver.minimize(T=1000.0, feed=feed)
        assert result.success

        # Reconstruct element moles from equilibrium
        phase = ideal_binary_phase()
        x = result.phase_compositions["liquid"]
        n_total = result.phase_amounts["liquid"]
        n_moles = x * n_total

        elem_out = {}
        for sp, n in zip(phase.species, n_moles):
            for el, count in sp.stoichiometry.items():
                elem_out[el] = elem_out.get(el, 0.0) + count * n

        for el, n_in in feed.items():
            np.testing.assert_allclose(
                elem_out.get(el, 0.0), n_in, rtol=1e-4,
                err_msg=f"Mass not conserved for element {el}"
            )

    def test_ternary_element_conservation(self):
        feed = {"La": 0.5, "Ce": 0.3, "Nd": 0.2}
        solver = GibbsSolver([ideal_ternary_phase()])
        result = solver.minimize(T=1200.0, feed=feed)
        assert result.success

        phase = ideal_ternary_phase()
        x = result.phase_compositions["liquid"]
        n_total = result.phase_amounts["liquid"]
        n_moles = x * n_total

        elem_out = {}
        for sp, n in zip(phase.species, n_moles):
            for el, count in sp.stoichiometry.items():
                elem_out[el] = elem_out.get(el, 0.0) + count * n

        for el, n_in in feed.items():
            np.testing.assert_allclose(
                elem_out.get(el, 0.0), n_in, rtol=1e-4
            )


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_no_negative_mole_amounts(self):
        solver = GibbsSolver([ideal_binary_phase()])
        result = solver.minimize(T=800.0, feed={"La": 1.0, "Ce": 1.0})
        for phase_name, n in result.phase_amounts.items():
            assert n >= 0, f"Negative moles in phase {phase_name}: {n}"

    def test_no_negative_mole_fractions(self):
        solver = GibbsSolver([ideal_binary_phase()])
        result = solver.minimize(T=800.0, feed={"La": 1.0, "Ce": 1.0})
        for phase_name, x in result.phase_compositions.items():
            assert (x >= 0).all(), f"Negative mole fractions in {phase_name}: {x}"


# ---------------------------------------------------------------------------
# Energy minimization
# ---------------------------------------------------------------------------


class TestEnergyMinimization:
    def test_g_total_le_initial_guess(self):
        """Optimizer must return G ≤ G of initial point (all-equal distribution)."""
        phase = ideal_binary_phase()
        solver = GibbsSolver([phase])
        feed = {"La": 1.0, "Ce": 1.0}
        T = 1000.0

        # G at equal composition (x = 0.5 each)
        x_eq = np.array([0.5, 0.5])
        mu = phase.chemical_potential(T, x_eq)
        G_initial = np.dot(np.array(list(feed.values())), mu)

        result = solver.minimize(T=T, feed=feed)
        assert result.success
        assert result.G_total <= G_initial + 1e-6  # tolerance for numerics

    def test_mole_fractions_sum_to_one(self):
        solver = GibbsSolver([ideal_binary_phase()])
        result = solver.minimize(T=1000.0, feed={"La": 1.0, "Ce": 1.0})
        for name, x in result.phase_compositions.items():
            if result.phase_amounts[name] > 1e-10:
                np.testing.assert_allclose(
                    x.sum(), 1.0, atol=1e-5,
                    err_msg=f"Mole fractions don't sum to 1 in {name}"
                )


# ---------------------------------------------------------------------------
# Phase diagram physical constraints
# ---------------------------------------------------------------------------


class TestPhaseDiagramConstraints:
    def test_ideal_gmix_negative_everywhere(self):
        """Ideal G_mix must be ≤ 0 for all interior compositions."""
        pd_gen = PhaseDiagram(ideal_binary_phase(), n_grid=100)
        result = pd_gen.binary(T=1000.0)
        # Exclude pure endpoints (x=0, x=1) where G_mix = 0
        interior = result.G_mix[1:-1]
        assert (interior <= 0).all(), "Ideal G_mix must be ≤ 0 in interior"

    def test_gmix_zero_at_pure_endpoints(self):
        """G_mix = 0 at pure component endpoints for ideal solution."""
        pd_gen = PhaseDiagram(ideal_binary_phase(), n_grid=200)
        result = pd_gen.binary(T=1000.0)
        np.testing.assert_allclose(result.G_mix[0], 0.0, atol=1e-3)
        np.testing.assert_allclose(result.G_mix[-1], 0.0, atol=1e-3)

    def test_ternary_gmix_shape(self):
        """Ternary result must have compositions summing to 1."""
        pd_gen = PhaseDiagram(ideal_ternary_phase(), n_grid=20)
        result = pd_gen.ternary(T=1000.0)
        sums = result.compositions.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Model constraint tests
# ---------------------------------------------------------------------------


class TestRegressionConstraints:
    def test_fit_predict_shape(self):
        T, X, G = make_synthetic_data(n=80, n_comp=2)
        model = NonlinearThermodynamicRegressor(n_components=2)
        model.fit(T, X, G)
        G_pred = model.predict(T, X)
        assert G_pred.shape == G.shape

    def test_prediction_not_nan(self):
        T, X, G = make_synthetic_data(n=80, n_comp=2)
        model = NonlinearThermodynamicRegressor(n_components=2)
        model.fit(T, X, G)
        G_pred = model.predict(T, X)
        assert not np.any(np.isnan(G_pred))

    def test_unfitted_model_raises(self):
        model = NonlinearThermodynamicRegressor(n_components=2)
        T, X, _ = make_synthetic_data(n=5, n_comp=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(T, X)


class TestGPRConstraints:
    def test_std_positive_on_unseen_data(self):
        T, X, G = make_synthetic_data(n=30, n_comp=2)
        features = np.column_stack([T.reshape(-1, 1), X])
        model = GaussianProcessThermo()
        model.fit(features, G)

        T_new, X_new, _ = make_synthetic_data(n=10, n_comp=2, seed=99)
        features_new = np.column_stack([T_new.reshape(-1, 1), X_new])
        pred = model.predict(features_new)
        assert (pred.std > 0).all(), "GPR std must be positive for unseen data"

    def test_gpr_mean_finite(self):
        T, X, G = make_synthetic_data(n=30, n_comp=2)
        features = np.column_stack([T.reshape(-1, 1), X])
        model = GaussianProcessThermo()
        model.fit(features, G)
        pred = model.predict(features)
        assert np.all(np.isfinite(pred.mean))


class TestEnsembleConstraints:
    @pytest.mark.parametrize("model_name", ["random_forest", "gradient_boosting"])
    def test_predictions_finite(self, model_name):
        T, X, G = make_synthetic_data(n=60, n_comp=2)
        features = np.column_stack([T.reshape(-1, 1), X])
        model = EnsembleThermo(model=model_name)
        model.fit(features, G)
        pred = model.predict(features)
        assert np.all(np.isfinite(pred.mean))

    def test_feature_importances_sum_to_one(self):
        T, X, G = make_synthetic_data(n=60, n_comp=2)
        features = np.column_stack([T.reshape(-1, 1), X])
        model = EnsembleThermo(model="random_forest")
        model.fit(features, G, feature_names=["T", "x_A", "x_B"])
        imp = model.feature_importances()
        np.testing.assert_allclose(sum(imp.values()), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Data pipeline constraints
# ---------------------------------------------------------------------------


class TestPipelineConstraints:
    def _make_dataframe(self, n: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        T = rng.uniform(600, 1400, n)
        raw = rng.exponential(1.0, (n, 2))
        X = raw / raw.sum(axis=1, keepdims=True)
        x_safe = np.clip(X, 1e-15, 1.0)
        G = R * T * np.sum(x_safe * np.log(x_safe), axis=1)
        return pd.DataFrame({"T_K": T, "La_x": X[:, 0], "Ce_x": X[:, 1], "G_mix": G})

    def test_output_shapes_consistent(self):
        cfg = PipelineConfig(elements=["La", "Ce"])
        pipeline = DataPipeline(cfg)
        df = self._make_dataframe(n=50)
        out = pipeline.run_from_dataframe(df)
        assert out.X_train.shape[0] == out.y_train.shape[0]
        assert out.X_test.shape[0] == out.y_test.shape[0]
        assert out.X_train.shape[1] == out.X_test.shape[1] == len(out.feature_names)

    def test_no_nan_in_output(self):
        cfg = PipelineConfig(elements=["La", "Ce"])
        pipeline = DataPipeline(cfg)
        df = self._make_dataframe(n=50)
        out = pipeline.run_from_dataframe(df)
        assert not np.any(np.isnan(out.X_train))
        assert not np.any(np.isnan(out.y_train))

    def test_train_test_sizes(self):
        cfg = PipelineConfig(elements=["La", "Ce"], test_fraction=0.2)
        pipeline = DataPipeline(cfg)
        df = self._make_dataframe(n=100)
        out = pipeline.run_from_dataframe(df)
        total = out.X_train.shape[0] + out.X_test.shape[0]
        assert total == 100

    def test_invalid_mole_fractions_raise(self):
        cfg = PipelineConfig(elements=["La", "Ce"])
        pipeline = DataPipeline(cfg)
        df = self._make_dataframe(n=10)
        df.loc[0, "La_x"] = 1.5  # invalid
        with pytest.raises(ValueError, match="Mole fractions out of"):
            pipeline.run_from_dataframe(df)

    def test_negative_temperature_raises(self):
        cfg = PipelineConfig(elements=["La", "Ce"])
        pipeline = DataPipeline(cfg)
        df = self._make_dataframe(n=10)
        df.loc[0, "T_K"] = -300.0
        with pytest.raises(ValueError, match="Temperature must be positive"):
            pipeline.run_from_dataframe(df)
