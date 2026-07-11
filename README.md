# Algorithmic Quant Engine

**Algorithmic Quant Engine** is a full-stack quantitative finance platform for real-time market simulation, stochastic forecasting, portfolio optimization, and fuzzy-logic risk analysis. Built on a decoupled **Model–View–Controller** architecture, it combines a custom NumPy/SciPy mathematical core with a Flask REST API and a live financial dashboard.

The system runs statistical estimation, stochastic differential equations, fuzzy inference, and constrained optimization as a single analytical pipeline — against either a synthetic simulated market or real historical data pulled from Yahoo Finance.

> ⚠️ **Disclaimer:** Default mode uses synthetic data structurally modeled after a high-frequency trip/tick schema. Real-data mode (documented below) is free and opt-in via `yfinance`. Nothing here connects to a live exchange or brokerage, and nothing here is investment advice.

---

## Table of Contents

- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Mathematical Engine](#mathematical-engine)
- [Mamdani Fuzzy Inference System](#mamdani-fuzzy-inference-system)
- [Numerical Stability](#numerical-stability)
- [REST API](#rest-api)
- [Dashboard](#dashboard)
- [Technology Stack](#technology-stack)
- [Installation & Running](#installation--running)
- [Real Market Data Mode (Optional, Free)](#real-market-data-mode-optional-free)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Features

- Real-time quantitative analysis via a polling dashboard (4s interval)
- Maximum Likelihood Estimation (MLE) of drift/volatility under GBM
- Euler–Maruyama stochastic forecasting (200-path Monte Carlo)
- Mamdani fuzzy inference system for a composite 0–100 risk score
- Ridge regression solved from scratch in closed form (no `sklearn.fit`)
- Constrained portfolio optimization via SciPy SLSQP
- Multi-threaded Flask REST API (bounded `ThreadPoolExecutor`)
- Optional real market data mode via `yfinance` (free, no API key)
- Interactive Chart.js dashboard: SDE fan chart, allocation pie, live ticker, stock price/volume
- Clean MVC separation — the math core has zero HTTP awareness
- Dark institutional-terminal UI with a custom fuzzy-risk gauge

---

## Pipeline Architecture

Every call to the analysis endpoint runs the same deterministic five-stage pipeline:

```
                  Dataset (synthetic or real)
                              |
                              v
            +-------------------------------------+
            | 1. Maximum Likelihood Estimation     |
            +-------------------------------------+
                              |
                 mu (Drift) and sigma (Volatility)
                              |
                              v
            +-------------------------------------+
            | 2. Euler-Maruyama SDE Forecast        |
            +-------------------------------------+
                              |
                  Multi-path price simulation
                              |
                              v
            +-------------------------------------+
            | 3. Closed-Form Ridge Regression        |
            +-------------------------------------+
                              |
                     Fitted coefficients
                              |
                              v
            +-------------------------------------+
            | 4. Portfolio Optimization (SLSQP)       |
            +-------------------------------------+
                              |
                     Minimum-variance weights
                              |
                              v
            +-------------------------------------+
            | 5. Mamdani Fuzzy Risk Inference         |
            +-------------------------------------+
                              |
                       Risk Asset Score (0-100)
```

All five stages run inside a single `run_full_analysis()` call and return as one JSON payload — see [REST API](#rest-api).

---

## Project Structure

```
algorithmic_quant_engine/
│
├── app.py                     # Controller: Flask REST API, thread pool, logging, error handlers
├── quant_model.py               # Model: MLE, SDE solver, custom ridge, fuzzy logic, optimizer, real-data ingestion
├── requirements.txt              # Pinned dependencies
├── LICENSE                       # Apache License 2.0
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html                # Dashboard shell (KPI strip, chart canvases, ticker search, detail panels)
│
└── static/
    ├── css/
    │   └── style.css             # Dark trading-terminal theme, CSS custom properties, grid layout
    └── js/
        └── dashboard.js           # ES6 QuantDashboard class: polling, fetch, Chart.js rendering
```

### `app.py` — Controller
- Flask REST API and route management
- Pre-warms the engine once at boot (synthetic or real, via env var)
- Offloads compute-heavy requests to a bounded `ThreadPoolExecutor`
- Structured request/response logging with a per-request correlation ID
- JSON error handling for 400 / 404 / 405 / 500 / 502 / 504

### `quant_model.py` — Model Core
- Synthetic dataset generation (vectorized log random walk)
- Real market data ingestion (`yfinance`, optional)
- MLE estimator for GBM drift/volatility
- Euler–Maruyama SDE solver
- Closed-form ridge regression
- Mamdani fuzzy inference system
- SLSQP portfolio optimizer
- Zero Flask/HTTP awareness — pure math, independently testable

### `templates/index.html` + `static/css/style.css`
Dashboard layout, KPI cards, chart canvases, ticker search form, and a dark institutional-terminal visual theme (CSS custom properties, responsive grid, semicircular risk gauge).

### `static/js/dashboard.js`
Vanilla ES6 class-based engine: `fetch`-based polling, Chart.js rendering, DOM updates — no build step, no framework.

---

## Mathematical Engine

### 1. Synthetic Market Data Generation
**Algorithm:** vectorized log random walk with distance-linked heteroskedasticity.
**Purpose:** produces a realistic, non-stationary financial time series for testing and demos without any external dependency.
**Characteristics (default configuration — tunable via constructor args):**
- 125,000 observations by default
- 250 ms simulated tick interval
- Non-stationary, heteroskedastic volatility
- Approximately log-normal returns

### 2. Maximum Likelihood Estimation (MLE)
Assumes log-returns follow the standard GBM log-normal model:

```text
log(S(t+1) / S(t)) ~ N( (mu - 0.5*sigma^2)*dt , sigma^2*dt )
```

Closed-form estimators:

```text
sigma_hat^2 = Var(r) / dt
mu_hat      = Mean(r) / dt + 0.5 * sigma_hat^2
```

**Outputs:** drift (mu), volatility (sigma), mean log-return, observation count.

### 3. Euler–Maruyama Stochastic Solver
Discretizes the GBM SDE `dS = mu*S*dt + sigma*S*dW_t`:

```text
S(t+1) = S(t) + mu*S(t)*dt + sigma*S(t)*sqrt(dt)*Z,   Z ~ N(0,1)
```

- **Forecast horizon:** 252 trading days (configurable)
- **Paths:** 200 by default, vectorized draw of all Wiener increments
- **Safety:** hard floor `S >= 1e-8` prevents negative/undefined prices from an Euler overshoot; explicit finiteness check after simulation

### 4. Custom Closed-Form Ridge Regression
No `scikit-learn` fit calls — solved directly with raw NumPy matrix algebra:

```text
beta = (X^T*X + alpha*I)^-1 * X^T*y
```

- L2 regularization with intercept excluded from the penalty
- Features standardized (zero mean, unit variance) before fitting
- Automatic Moore–Penrose pseudo-inverse fallback on singular matrices
- Full dimensionality validation before any matrix inversion is attempted

### 5. Portfolio Optimization
**Method:** Sequential Least Squares Programming (SLSQP), via `scipy.optimize.minimize`, with an explicit analytical gradient for faster convergence.

```text
minimize    w^T * Sigma * w
subject to  sum(w) = 1
            0 <= w_i <= 1
```

**Asset universe:** either four synthetic distance-tier buckets (default) or a real multi-ticker covariance matrix (opt-in — see [Real Market Data Mode](#real-market-data-mode-optional-free)).
**Outputs:** portfolio weights, portfolio volatility, expected portfolio return, solver convergence status.

---

## Mamdani Fuzzy Inference System

The risk engine evaluates two crisp inputs — **not three** — through a full Mamdani pipeline:

**Inputs:** Volatility, Demand Density *(momentum is not currently a fuzzy input; see Future Improvements)*

**Membership functions:** triangular, three linguistic terms per input (Low / Medium / High)

**Rule base:** 9 rules, e.g.:
```
IF Volatility is HIGH AND Demand is LOW  -> Risk is MEDIUM
IF Volatility is HIGH AND Demand is HIGH -> Risk is HIGH
```

**Aggregation:** Mamdani AND = min; rule outputs combined via max

**Defuzzification:** centroid (center-of-gravity), computed via discretized numerical integration:

```text
             integral of x * mu(x) dx
Risk score = -------------------------
              integral of mu(x) dx
```

**Output:** Risk Asset Score, range `[0, 100]`.

---

## Numerical Stability

Fault-tolerance mechanisms built into the math core:

| Failure mode | Mitigation |
|---|---|
| Degenerate triangular membership span (`a == b` or `b == c`) | Treated as a step function instead of dividing by zero |
| Zero-area aggregated fuzzy output | Falls back to a neutral Risk Score of `50.0`, logged as a warning |
| Singular `(X^T*X + alpha*I)` matrix | Automatically switches to Moore–Penrose pseudo-inverse |
| Euler–Maruyama price overshoot below zero | Hard floor `S >= 1e-8` |
| Non-finite (NaN/Inf) SDE output | Explicit `np.isfinite` check raises a clear error rather than propagating silently |
| Real-data fetch failure (network/invalid ticker) | Automatic fallback to synthetic mode at boot — see below |

---

## REST API

The Flask application exposes the following endpoints. Note this differs from a per-metric-endpoint design — **all analytical outputs are returned together from one endpoint**, since they share a single upstream computation (`run_full_analysis()`), and splitting them would mean re-running the whole pipeline per metric.

### `GET /`
Serves the dashboard shell.

### `GET /api/compute-quant-matrix`
Runs the full five-stage pipeline and returns everything — MLE, SDE forecast, ridge coefficients, portfolio allocation, and fuzzy risk score — in one JSON payload.

| Query param | Type | Default | Description |
|---|---|---|---|
| `ridge_alpha` | float | `1.0` | L2 regularization strength |

```json
{
  "meta": { "n_records": 125000, "computation_time_ms": 34.76, "data_source": "synthetic" },
  "mle": { "mu": 0.0003, "sigma": 0.011, "n_observations": 124999 },
  "sde_forecast": { "mean_path": [], "p05_path": [], "p95_path": [], "sample_paths": [[]] },
  "ridge_regression": { "intercept": 2.58, "coefficients": {}, "r_squared": 0.0001 },
  "portfolio_allocation": { "weights": [0.25, 0.25, 0.25, 0.25], "portfolio_volatility": 0.00 },
  "fuzzy_risk": { "risk_score": 13.46, "volatility_memberships": {}, "demand_memberships": {} },
  "raw_series": { "timestamps_ms": [], "velocity": [], "distance": [], "price": [] }
}
```

Errors: `400` invalid `ridge_alpha`, `500` engine failure, `504` exceeded 30s compute budget.

### `GET /api/stock-chart`
Real OHLCV bar data for a single ticker (independent of the main pipeline's data mode).

| Query param | Type | Default |
|---|---|---|
| `ticker` | string | `AAPL` |
| `period` | string | `6mo` |
| `interval` | string | `1d` |

Errors: `400` malformed ticker, `502` Yahoo Finance unreachable/invalid ticker, `504` timeout.

### `GET /api/health`
Lightweight liveness probe, independent of the compute pool:

```json
{ "status": "OK", "engine_warm": true, "n_records": 125000, "compute_pool_max_workers": 4 }
```

---

## Dashboard

Frontend is intentionally dependency-light: vanilla ES6, Chart.js, no build step.

**Components:**
- Fuzzy-risk gauge (custom SVG, semicircular, needle-based)
- Drift / volatility / ridge-intercept KPI cards
- SDE forecast fan chart (mean path + 5th/95th percentile envelope + sample paths)
- Portfolio allocation doughnut chart
- Live raw velocity ticker
- Real stock price/volume chart with ticker search
- Ridge coefficient table, fuzzy membership bars, system stat list

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, `ThreadPoolExecutor` |
| Math Core | NumPy, SciPy (`optimize.minimize`, SLSQP), Pandas |
| Real Data (optional) | `yfinance` |
| Frontend | Vanilla ES6 JavaScript, Chart.js 4 |
| Styling | Hand-written CSS3, custom properties, CSS Grid |
| Fonts | Space Grotesk (display), JetBrains Mono (data) |

**Supported platforms:** Windows, Linux, macOS · **Requires:** Python 3.10+

---

## Installation & Running

```bash
# 1. Clone
git clone https://github.com/nandanmukherjee917/algorithmic_quant_engine.git
cd algorithmic_quant_engine

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Open `http://127.0.0.1:5000` in a browser. For production, run behind a real WSGI server:

```bash
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

---

## Real Market Data Mode (Optional, Free)

By default the engine runs entirely offline on synthetic data. Switch to real data with two environment variables — no API key, no signup, no cost:

```bash
export QUANT_DATA_MODE=real
export QUANT_TICKER=AAPL
export QUANT_PORTFOLIO_TICKERS=AAPL,MSFT,GOOGL,AMZN
python app.py
```

At boot, `warm_up(mode="real")` fetches real OHLCV data via `yfinance` and remaps it onto the engine's internal schema; `set_multi_asset_universe()` builds a genuine covariance matrix across the portfolio tickers. **If the fetch fails for any reason (no network, invalid ticker), the engine automatically falls back to synthetic mode and logs a warning** — it never fails to boot because of a network hiccup. Check `meta.data_source` in the API response to confirm which mode is actually active.

---

## Known Limitations

Being direct about the current state:

- **Frontend rendering is not browser-verified.** Backend and API were tested end-to-end via direct execution and `curl`; the dashboard's charts, gauge, and layout have not been confirmed in an actual browser session.
- **Real-data mode's success path is untested against live Yahoo Finance** in the environment this was built in (no general internet egress). The failure/fallback path was verified; the happy path was not.
- **Single-process dev server** — fine for demos, use `gunicorn`/`uwsgi` behind a reverse proxy for anything real.
- **No automated test suite** — verification has been manual (`curl` + direct engine calls), not `pytest`-based.
- **Synthetic portfolio buckets can converge to near-equal weights**, since all four distance tiers derive from correlated slices of one price path — mathematically correct, just not visually dramatic. Real multi-asset mode does not have this issue.

---

## Future Improvements

- Monte Carlo option pricing (Black–Scholes benchmark)
- Value at Risk (VaR) / Conditional VaR
- GARCH volatility estimation
- LSTM-based forecasting
- Reinforcement-learning portfolio management
- Multi-asset correlation engine beyond the current 4-tier/4-ticker setup
- Add momentum as a third fuzzy input (currently only volatility + demand density)
- True OHLC candlestick charting via `chartjs-chart-financial` (free, via jsdelivr)
- WebSocket streaming instead of polling
- Docker + Kubernetes deployment
- Authentication and PostgreSQL persistence
- `pytest` suite (e.g. cross-checking the custom ridge solver against `sklearn.linear_model.Ridge`)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE). You may use, modify, and distribute this project — including commercially — provided you retain the copyright notice and state any changes you make. See [`LICENSE`](LICENSE) for the full text.

---

## Acknowledgements

This project draws on concepts from quantitative finance, stochastic calculus, optimization theory, and fuzzy systems, including Geometric Brownian Motion, Maximum Likelihood Estimation, the Euler–Maruyama method, Ridge Regression, Mamdani Fuzzy Inference, Sequential Least Squares Programming, and Modern Portfolio Theory.

---

## Author

**Nandan Mukherjee**
Artificial Intelligence & Machine Learning Undergraduate
Quantitative Finance & Theoretical Physics Enthusiast · Aspiring Machine Learning Engineer

This project integrates stochastic calculus, optimization theory, statistical estimation, fuzzy inference, and modern web technologies into a single quantitative finance platform built with Python, Flask, and MVC architecture.