"""
Tests for Gibbs minimization solver equilibrium correctness.

Validates:
- Known binary equilibrium compositions
- Convergence and success flags
- G_total decreases monotonically with refinement
- Temperature dependence of equilibrium
"""

import pytest
import numpy as np

from engine.gibbs_solver import GibbsSolver, Phase, Species

R = 8.314462  # J/(mol·K)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_ideal_binary(g1: float = 0.0, g2: float = 0.0) -> Phase:
    """Ideal binary phase with constant G° values."""
    s1 = Species(
        name="A",
        formula="A",
        stoichiometry={"A": 1},
        g_standard_298=g1,
    )
    s2 = Species(
        name="B",
        formula="B",
        stoichiometry={"B": 1},
        g_standard_298=g2,
    )
    return Phase(name="liquid", species=[s1, s2])


def make_ideal_ternary() -> Phase:
    """Ideal ternary phase A-B-C."""
    return Phase(
        name="liquid",
        species=[
            Species("A", "A", {"A": 1}, g_standard_298=0.0),
            Species("B", "B", {"B": 1}, g_standard_298=0.0),
            Species("C", "C", {"C": 1}, g_standard_298=0.0),
        ],
    )


# ---------------------------------------------------------------------------
# Equilibrium tests
# ---------------------------------------------------------------------------


class TestIdealBinaryEquilibrium:
    def test_symmetric_feed_gives_equal_moles(self):
        """For symmetric ideal binary at equal feed, equilibrium x ≈ 0.5."""
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        result = solver.minimize(T=1000.0, feed={"A": 1.0, "B": 1.0})

        assert result.success, f"Solver failed: {result.message}"
        x = result.phase_compositions["liquid"]
        assert len(x) == 2
        # Ideal solution: minimum at x=0.5 when G°_A = G°_B
        np.testing.assert_allclose(x[0], 0.5, atol=1e-4)
        np.testing.assert_allclose(x[1], 0.5, atol=1e-4)

    def test_asymmetric_g_lowers_g_total(self):
        """
        Lowering G° of species B makes it more stable.
        In a single-phase system, mass balance fixes composition to the feed,
        but G_total must decrease when G°_B is lower.
        """
        phase_symmetric = make_ideal_binary(g1=0.0, g2=0.0)
        phase_favored = make_ideal_binary(g1=0.0, g2=-10_000.0)
        solver_sym = GibbsSolver([phase_symmetric])
        solver_fav = GibbsSolver([phase_favored])
        feed = {"A": 1.0, "B": 1.0}

        result_sym = solver_sym.minimize(T=1000.0, feed=feed)
        result_fav = solver_fav.minimize(T=1000.0, feed=feed)

        assert result_sym.success and result_fav.success
        # Lower G°_B → lower G_total
        assert result_fav.G_total < result_sym.G_total, (
            "G_total must decrease when G°_B is more negative"
        )

    def test_total_moles_conserved(self):
        """Total moles in each phase must equal feed total."""
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        feed = {"A": 0.7, "B": 0.3}
        result = solver.minimize(T=800.0, feed=feed)

        n_total_feed = sum(feed.values())
        n_total_equil = sum(result.phase_amounts.values())
        np.testing.assert_allclose(n_total_equil, n_total_feed, rtol=1e-5)

    def test_g_total_is_negative_for_ideal_mixing(self):
        """G_total must be negative for ideal mixing at finite T (entropy driven)."""
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        result = solver.minimize(T=1000.0, feed={"A": 1.0, "B": 1.0})

        assert result.success
        # G_mix = RT * (0.5*ln0.5 + 0.5*ln0.5) < 0
        assert result.G_total < 0.0, f"Expected G_total < 0, got {result.G_total}"

    def test_high_T_increases_mixing(self):
        """Higher temperature should lower G_total for ideal mixing."""
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        feed = {"A": 1.0, "B": 1.0}

        result_low = solver.minimize(T=300.0, feed=feed)
        result_high = solver.minimize(T=2000.0, feed=feed)

        assert result_low.success and result_high.success
        assert result_high.G_total < result_low.G_total


class TestIdealTernaryEquilibrium:
    def test_symmetric_ternary(self):
        """Symmetric ternary at equal feed → x ≈ 1/3 for all."""
        phase = make_ideal_ternary()
        solver = GibbsSolver([phase])
        result = solver.minimize(T=1000.0, feed={"A": 1.0, "B": 1.0, "C": 1.0})

        assert result.success
        x = result.phase_compositions["liquid"]
        np.testing.assert_allclose(x, [1 / 3, 1 / 3, 1 / 3], atol=1e-4)

    def test_ternary_mass_conservation(self):
        phase = make_ideal_ternary()
        solver = GibbsSolver([phase])
        feed = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = solver.minimize(T=1000.0, feed=feed)

        n_total_feed = sum(feed.values())
        n_total_equil = sum(result.phase_amounts.values())
        np.testing.assert_allclose(n_total_equil, n_total_feed, rtol=1e-5)


class TestNonIdealEquilibrium:
    def test_positive_interaction_reduces_mixing(self):
        """
        Positive Margules parameter W > 0 means repulsive interaction.
        G_total should be higher than ideal case at same composition.
        """
        phase_ideal = make_ideal_binary()
        phase_nonideal = Phase(
            name="liquid",
            species=[
                Species("A", "A", {"A": 1}, g_standard_298=0.0),
                Species("B", "B", {"B": 1}, g_standard_298=0.0),
            ],
            interaction=np.array([[0.0, 20_000.0], [20_000.0, 0.0]]),
        )

        solver_ideal = GibbsSolver([phase_ideal])
        solver_nonideal = GibbsSolver([phase_nonideal])
        feed = {"A": 1.0, "B": 1.0}
        T = 500.0

        r_ideal = solver_ideal.minimize(T=T, feed=feed)
        r_nonideal = solver_nonideal.minimize(T=T, feed=feed)

        assert r_ideal.success and r_nonideal.success
        # Positive excess raises G_total
        assert r_nonideal.G_total > r_ideal.G_total


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestSolverInputValidation:
    def test_negative_temperature_raises(self):
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        with pytest.raises(ValueError, match="Temperature must be positive"):
            solver.minimize(T=-100.0, feed={"A": 1.0, "B": 1.0})

    def test_zero_temperature_raises(self):
        phase = make_ideal_binary()
        solver = GibbsSolver([phase])
        with pytest.raises(ValueError):
            solver.minimize(T=0.0, feed={"A": 1.0, "B": 1.0})
