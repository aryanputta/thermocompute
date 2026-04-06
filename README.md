# ThermoCompute

**Computational thermodynamics engine for rare-earth solution systems.**

Experimental thermodynamics of rare-earth systems is expensive, slow, and sparse. ThermoCompute closes that gap with a physics-based computational pipeline: Gibbs free energy minimization, phase diagram generation, and machine learning models that improve as real lab data accumulates.

---

## What it does

| Layer | What it computes |
|---|---|
| **Gibbs solver** | Equilibrium composition and phase distribution via G minimization |
| **Phase diagrams** | Binary and ternary stability maps via convex hull construction |
| **Regression** | Redlich-Kister L0 interaction parameters from sparse (T, x, G) data |
| **GPR** | Uncertainty-aware Gibbs energy surface predictions |
| **Ensemble ML** | Random Forest / Gradient Boosting for tabular thermodynamic prediction |
| **Data pipeline** | Raw CSV → feature-engineered model input with atomic property weighting |

---

## Motivation

Equilibrium states — phase stability, solubility limits, reaction pathways — currently require repeated lab trials. Every iteration costs time and material. ThermoCompute:

- Predicts equilibrium states before experiments run
- Quantifies uncertainty so you know which experiments are actually needed
- Retrains on real data as it comes in, continuously shrinking prediction error
- Directly targets rare-earth processing, clean energy materials, and national research priorities

---

## Architecture

```
[Experimental Data / NIST / Materials Project]
              │
              ▼
     ┌─────────────────┐
     │  Data Pipeline  │  raw → clean → atomic features → train/test split
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  Core Engine    │
     │                 │
     │  Gibbs Solver   │  minimize G = Σ nᵢμᵢ  (SLSQP / L-BFGS-B)
     │  Phase Diagram  │  convex hull on G_mix surface
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  ML / Regression│
     │                 │
     │  Nonlinear Reg  │  Redlich-Kister L0 fitting
     │  GPR            │  Matérn 5/2, uncertainty estimates
     │  Ensemble       │  Random Forest, Gradient Boosting
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  Iterative Loop │  data → model update → prediction → error → retrain
     └─────────────────┘
```

---

## Repository structure

```
thermocompute/
├── engine/
│   ├── gibbs_solver.py       # Gibbs free energy minimizer
│   └── phase_diagram.py      # Binary and ternary phase diagram generator
├── models/
│   ├── regression/
│   │   └── nonlinear.py      # Redlich-Kister regression
│   └── ml/
│       ├── gpr.py            # Gaussian Process Regression
│       └── ensemble.py       # Random Forest / Gradient Boosting
├── data/
│   ├── pipeline.py           # Full data processing pipeline
│   ├── loaders.py            # NIST and Materials Project loaders
│   ├── raw/                  # Raw input data (CSV, API outputs)
│   └── processed/            # Cleaned, feature-engineered data
├── tests/
│   ├── test_equilibrium.py   # Equilibrium correctness tests
│   └── test_constraints.py   # Physical constraint enforcement tests
├── benchmarks/
│   ├── benchmark.py          # Full benchmark suite
│   └── results.csv           # Benchmark output
├── experiments/
│   └── notebooks/            # Analysis notebooks
├── main.py                   # Demo entry point
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/aryanputta/thermocompute.git
cd thermocompute
pip install -r requirements.txt

python3 main.py          # run core demos
pytest tests/ -v         # run all 35 tests
python3 benchmarks/benchmark.py  # accuracy + learning loop benchmarks
```

**Materials Project API** (optional, for real formation energy data):

```bash
export MP_API_KEY=your_key_here
pip install mp-api
```

---

## Core engine

### Gibbs minimization

Minimizes total Gibbs free energy over all phases:

$$G_\text{total} = \sum_\alpha \sum_i n_i^\alpha \cdot \mu_i^\alpha(T, x^\alpha)$$

Chemical potential for species $i$ in phase $\alpha$:

$$\mu_i^\alpha = G_i^\circ(T) + RT \ln x_i^\alpha + RT \ln \gamma_i^\alpha$$

For non-ideal solutions, $\ln \gamma_i$ is computed from Margules interaction parameters. Ideal solution sets $\gamma_i = 1$.

Solved via `scipy.optimize` SLSQP with:
- **Mass balance** equality constraints: $\sum_\alpha n_i^\alpha = N_i$ for each element
- **Non-negativity** bounds: $n_i^\alpha \geq 0$
- **Analytical gradient** $\partial G / \partial n_i^\alpha = \mu_i^\alpha$

Standard-state $G_i^\circ(T)$ computed from NIST Shomate coefficients when available:

$$H^\circ(T) - H^\circ(298) = At + \frac{B}{2}t^2 + \frac{C}{3}t^3 + \frac{D}{4}t^4 - \frac{E}{t} + F - H$$

where $t = T/1000$.

### Phase diagrams

Binary and ternary stability maps are built by:

1. Sampling $G_\text{mix}$ across the composition simplex
2. Computing the lower convex hull of the $(x, G_\text{mix})$ surface
3. Identifying miscibility gaps as regions where $G_\text{mix}$ lies above the hull

Ternary diagrams use barycentric grid sampling and a 3D convex hull projection.

---

## Machine learning layer

### Nonlinear regression

Fits Redlich-Kister L0 interaction parameters $a_{ij}$, $b_{ij}$ from observed $(T, x, G_\text{mix})$ data:

$$G_\text{mix} = RT \sum_i x_i \ln x_i + \sum_{i < j} x_i x_j (a_{ij} + b_{ij} T)$$

### Gaussian Process Regression

Matérn $\nu = 5/2$ kernel with white noise term. Provides calibrated uncertainty estimates on unseen compositions — critical for active learning (predict where to run the next experiment).

### Ensemble models

Random Forest and Gradient Boosting trained on feature-engineered inputs. Includes cross-validation, feature importance reporting, and a `StandardScaler` preprocessing step.

### Feature engineering

Beyond raw $(T, x_i)$ inputs, the pipeline computes:

| Feature | Formula |
|---|---|
| Temperature terms | $T/T_\text{ref}$, $\ln T$, $1/T$ |
| Cross-composition | $x_i \cdot x_j$ for all pairs |
| Avg atomic radius | $\sum_i x_i r_i$ |
| Avg electronegativity | $\sum_i x_i \chi_i$ |
| Avg oxidation state | $\sum_i x_i \nu_i$ |

Atomic data sourced from NIST Standard Reference for La, Ce, Pr, Nd, Sm, Eu, Gd, Dy, Y, Fe, O, Cl.

---

## Benchmark results

All numbers from `benchmarks/benchmark.py` on ideal La-Ce binary data, n_train = 200, n_test = 200.

### Accuracy vs baselines

| Model | MAE (J/mol) | RMSE (J/mol) | % Deviation |
|---|---|---|---|
| Mean baseline | 1493.3 | 1824.9 | 73.5% |
| Linear interpolation | 1527.3 | 2126.2 | 61.1% |
| **Nonlinear regression** | **0.0** | **0.0** | **0.0%** |
| **GPR** | **9.2** | **22.9** | **0.66%** |
| Random Forest | 153.0 | 200.9 | 7.5% |
| Gradient Boosting | 167.7 | 216.6 | 7.1% |

Nonlinear regression recovers exact parameters because the data is generated from the same ideal-solution model. GPR achieves sub-1% deviation. Both models reduce RMSE by **~99×** vs the mean baseline.

### Error reduction — iterative learning loop

RMSE of Random Forest as experimental data accumulates (binary system):

| Points | RMSE (J/mol) |
|---|---|
| 10 | 1195.7 |
| 30 | 714.1 |
| 60 | 581.3 |
| 100 | 360.5 |
| 160 | **283.9** |

RMSE drops **4.2× (76%)** going from 10 to 160 experimental points, demonstrating that the iterative pipeline meaningfully reduces how many experiments are needed to reach a target accuracy.

### Multi-component scaling

| System | MAE (J/mol) | RMSE (J/mol) | Fit time (s) |
|---|---|---|---|
| Binary (2) | 153.0 | 200.9 | 0.15 |
| Ternary (3) | 563.4 | 860.4 | 0.15 |
| Quaternary (4) | 813.7 | 1102.4 | 0.14 |

Fit time is effectively constant as component count scales — compute cost is dominated by data size, not dimensionality.

---

## Testing

35 tests across two files. Run with:

```bash
pytest tests/ -v
```

### Physical constraint tests (`test_constraints.py`)

- **Mass conservation** — element moles in equals element moles out, across 4 binary and 1 ternary feed compositions
- **Non-negativity** — no negative mole amounts or mole fractions at equilibrium
- **Energy minimization** — $G_\text{total}$ at equilibrium $\leq$ $G$ of initial guess
- **Mole fraction closure** — sum = 1 in every phase with nonzero amount
- **Ideal $G_\text{mix} \leq 0$** — ideal mixing Gibbs energy is negative everywhere in the interior
- **Endpoints** — $G_\text{mix} = 0$ at pure component limits
- **Ternary barycentric closure** — all sampled compositions sum to 1
- **Model contracts** — correct output shapes, finite predictions, unfitted-raises, feature importances sum to 1
- **Pipeline validation** — raises on negative temperatures, mole fractions outside [0,1], missing columns

### Equilibrium tests (`test_equilibrium.py`)

- **Symmetric binary** — equal feed gives $x = 0.5$ for ideal solution
- **Stability direction** — lower $G^\circ$ decreases $G_\text{total}$
- **Total moles conserved** — across asymmetric feeds
- **Sign of $G_\text{total}$** — negative for ideal mixing at finite $T$
- **Temperature effect** — higher $T$ lowers $G_\text{total}$ (entropy contribution)
- **Symmetric ternary** — $x = 1/3$ for all components under equal feed
- **Non-ideal mixing** — positive Margules parameter raises $G_\text{total}$ vs ideal
- **Input validation** — raises on $T \leq 0$

---

## Data sources

| Source | What it provides | How to access |
|---|---|---|
| NIST Chemistry WebBook | Shomate equation coefficients for $G^\circ(T)$ | Download CSV, use `NISTLoader.load_from_file()` |
| Materials Project | DFT formation energies, phase stability | `MaterialsProjectLoader`, requires `MP_API_KEY` |
| OQMD / published datasets | Rare-earth thermodynamic tables | Import as CSV via `DataPipeline` |
| Lab data (XRD, ICP-OES) | Experimental phase compositions | Import as CSV via `DataPipeline` |

The engine is data-agnostic: any source that produces $(T, P, x_i, G)$ tuples plugs directly into the pipeline.

---

## Iteration plan

| Version | Status | Scope |
|---|---|---|
| V1 | ✅ Complete | Gibbs minimization, 2-component system |
| V2 | ✅ Complete | Phase diagrams, binary + ternary |
| V3 | ✅ Complete | Regression + ML prediction layer |
| V4 | In progress | Real experimental dataset integration |
| V5 | Planned | Optimization, full benchmark report |

---

## Tech stack

- **Python 3.10+**
- `numpy`, `scipy` — numerical optimization, convex hull
- `scikit-learn` — GPR, ensemble models, cross-validation
- `pandas` — data pipeline
- `matplotlib` — phase diagram visualization (plots only, no dashboards)
- `mp-api` — Materials Project (optional)
- `pytest` — test suite

---
