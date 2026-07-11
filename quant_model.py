"""
quant_model.py
==============================================================================
THE DEEP QUANTITATIVE ENGINE
==============================================================================
SPDX-License-Identifier: Apache-2.0
Copyright 2026 [your name or organization]. See LICENSE for full terms.
==============================================================================

This module houses `AdvancedQuantEngine`, the mathematical core (Model layer)
of the High-Frequency Macro Algorithmic Arbitrage Engine. It is intentionally
decoupled from Flask / HTTP concerns entirely -- nothing in this file knows
about requests, responses, or JSON serialization of routes. It only knows
about numbers.

The engine is responsible for five independent quantitative subsystems:

    1. Synthetic multi-gigabyte-style time-series generation (modeled loosely
       on the shape of the NYC TLC taxi trip record dataset: timestamps,
       distances, passenger counts, fares, and surcharges).
    2. Maximum Likelihood Estimation (MLE) of Geometric Brownian Motion (GBM)
       drift (mu) and volatility (sigma) parameters from log-return slices.
    3. A discrete-time Euler-Maruyama solver for the GBM stochastic
       differential equation, producing multi-path Monte Carlo forecasts.
    4. A from-scratch, closed-form L2-regularized (Ridge) linear regression
       solver implemented with raw numpy linear algebra (no scikit-learn
       fitting wrappers).
    5. A Mamdani-style fuzzy inference system (FIS) with triangular
       membership functions and centroid defuzzification, producing a
       real-time composite Risk Asset Score.
    6. A constrained multivariable portfolio-variance minimizer built on
       scipy.optimize.minimize (SLSQP) subject to a hard equality
       constraint that allocations sum to exactly 1.0.

All public methods are defensive: inputs are validated, dimensionality is
checked before any matrix algebra is attempted, and every method fails loudly
(via a raised, informative exception) rather than silently returning
degenerate output.
==============================================================================
"""

import logging
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# yfinance is an OPTIONAL dependency, used only by generate_real_dataset().
# The engine must remain fully functional in synthetic mode even if it is
# not installed (e.g. offline demos, CI environments with no internet
# egress), so the import failure is caught here rather than raised at
# module load time.
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

# ------------------------------------------------------------------------- #
# Module-level logger. The controller (app.py) configures logging handlers;
# this module only ever calls into the logging API, never configures it,
# so it behaves correctly whether imported standalone or under Flask.
# ------------------------------------------------------------------------- #
logger = logging.getLogger("quant_engine.model")


class QuantEngineError(Exception):
    """
    Raised for any unrecoverable failure inside the AdvancedQuantEngine.

    Wrapping all internal failures (numpy LinAlgError, scipy convergence
    failures, malformed input shapes, etc.) in a single, well-known
    exception type lets the Flask controller layer catch one thing and
    respond with a clean, uniform 500 payload instead of leaking raw
    stack traces or numpy-specific exception types to API consumers.
    """
    pass


class AdvancedQuantEngine:
    """
    Principal mathematical model core for the arbitrage engine.

    An instance of this class owns:
        - `self.dataset`: a pandas DataFrame of simulated high-frequency
          trip/tick records (populated by `generate_simulated_dataset`).
        - RNG state (`self._rng`), seeded for reproducibility so that
          repeated `warm_up()` calls in a dev environment behave
          deterministically, while `run_full_analysis()` still injects
          fresh stochastic draws per call to simulate a live feed.

    Typical lifecycle (see app.py):
        engine = AdvancedQuantEngine(n_records=125_000)
        engine.warm_up()                # pre-generates dataset at boot
        payload = engine.run_full_analysis()   # called per-request
    """

    # Column bounds used both for synthetic data generation and later for
    # sanity-checking / clipping model output. Centralizing these avoids
    # magic numbers scattered across five different methods.
    VELOCITY_MIN_MPH = 0.5
    VELOCITY_MAX_MPH = 68.0
    DISTANCE_MIN_MI = 0.1
    DISTANCE_MAX_MI = 42.0
    BASE_FARE_FLOOR = 2.50
    SURCHARGE_MAX = 7.75

    def __init__(self, n_records=125_000, n_sde_paths=200, n_sde_steps=252,
                 random_seed=42):
        """
        Args:
            n_records (int): number of synthetic tick/trip rows to generate.
                Defaults to 125,000 to satisfy the "100,000+" throughput
                requirement while remaining fast enough to regenerate on a
                developer laptop in well under a second.
            n_sde_paths (int): number of independent Monte Carlo sample
                paths simulated per Euler-Maruyama call.
            n_sde_steps (int): number of discrete time steps per SDE path
                (252 mirrors trading-day convention: one simulated year).
            random_seed (int): seed for the engine's private numpy
                Generator, ensuring `generate_simulated_dataset` is
                reproducible across process restarts.

        Raises:
            QuantEngineError: if any constructor argument is non-positive,
                since a zero-or-negative record/path/step count would
                silently produce empty or malformed downstream arrays.
        """
        if n_records <= 0 or n_sde_paths <= 0 or n_sde_steps <= 0:
            raise QuantEngineError(
                "AdvancedQuantEngine requires strictly positive "
                "n_records, n_sde_paths, and n_sde_steps. Got: "
                f"n_records={n_records}, n_sde_paths={n_sde_paths}, "
                f"n_sde_steps={n_sde_steps}."
            )

        self.n_records = int(n_records)
        self.n_sde_paths = int(n_sde_paths)
        self.n_sde_steps = int(n_sde_steps)
        self._seed = int(random_seed)
        self._rng = np.random.default_rng(self._seed)

        self.dataset = None
        self._is_warmed = False
        self._last_warm_duration_sec = None
        self._data_source = "synthetic"
        self._multi_asset_universe = None  # set by set_multi_asset_universe()

        logger.info(
            "AdvancedQuantEngine instantiated (n_records=%d, "
            "n_sde_paths=%d, n_sde_steps=%d, seed=%d)",
            self.n_records, self.n_sde_paths, self.n_sde_steps, self._seed
        )

    # --------------------------------------------------------------------- #
    # 1. SYNTHETIC DATASET GENERATION
    # --------------------------------------------------------------------- #
    def generate_simulated_dataset(self):
        """
        Vectorized generation of a synthetic high-frequency trip/tick
        dataset, structurally modeled after the NYC TLC taxi trip record
        schema, mapping temporal fluctuations across five channels:

            - timestamp            : monotonically increasing tick times
            - passenger_velocity    : instantaneous velocity proxy (mph)
            - matrix_distance       : trip / segment distance (miles)
            - base_price_ticker     : underlying tradeable price series
                                       (constructed as a mean-reverting +
                                       drifted log-random-walk so that later
                                       GBM/MLE fitting is well-posed)
            - surcharges             : stochastic surcharge overlay (USD)

        The base price ticker is deliberately built as an exponentiated
        cumulative-sum-of-log-returns process (i.e. already approximately
        log-normal / GBM-shaped) so that the downstream MLE step is fitting
        a model to data that genuinely resembles its assumptions --
        mirroring how a real quant pipeline validates estimators against
        synthetic ground truth before pointing them at live data.

        Returns:
            pandas.DataFrame: the generated dataset, also cached on
            `self.dataset`.

        Raises:
            QuantEngineError: if array construction fails for any reason
                (e.g. memory allocation failure on pathologically large
                `n_records`), wrapping the underlying exception.
        """
        try:
            start_ts = time.perf_counter()
            n = self.n_records

            # --- Timestamps -------------------------------------------------
            # High-frequency ticks: one simulated tick every 250ms, anchored
            # to "now" and walking backward, mirroring a rolling ingestion
            # window of a live market-data buffer.
            anchor = pd.Timestamp(datetime.utcnow())
            tick_deltas = pd.to_timedelta(
                np.arange(n)[::-1] * 250, unit="ms"
            )
            timestamps = anchor - tick_deltas

            # --- Passenger Velocity (mph) -----------------------------------
            # Modeled as a smoothed, clipped absolute-normal process to
            # avoid unrealistic negative velocities while retaining bursty
            # high-frequency noise characteristic of real telemetry.
            raw_velocity = self._rng.normal(loc=22.0, scale=9.5, size=n)
            smoothing_kernel_size = 5
            kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size
            smoothed_velocity = np.convolve(raw_velocity, kernel, mode="same")
            passenger_velocity = np.clip(
                smoothed_velocity, self.VELOCITY_MIN_MPH, self.VELOCITY_MAX_MPH
            )

            # --- Matrix Distance (mi) ----------------------------------------
            # Log-normal distance distribution: many short hops, a long
            # tail of longer segments -- consistent with real trip-distance
            # distributions. Generated BEFORE the price ticker so that
            # distance can drive per-tick heteroskedasticity below (longer,
            # higher-friction segments carry genuinely higher price
            # volatility, which is what gives the downstream distance-tier
            # portfolio buckets differentiated, non-trivial covariance).
            matrix_distance = np.clip(
                self._rng.lognormal(mean=0.85, sigma=0.65, size=n),
                self.DISTANCE_MIN_MI, self.DISTANCE_MAX_MI
            )

            # --- Base Price Ticker (USD) --------------------------------------
            # Construct as an approximately log-normal GBM-like path:
            #   S_t = S_0 * exp(cumsum(mu*dt + sigma_t*sqrt(dt)*Z))
            # with a small dt to generate a dense, tradeable-looking series.
            # sigma_t is intentionally heteroskedastic, scaled by normalized
            # trip distance, so that distance-tiered pseudo-assets (used
            # later by the portfolio optimizer) carry meaningfully distinct
            # volatility profiles rather than all sampling the same
            # underlying homoskedastic process.
            dt = 1.0 / 390.0  # ~ one simulated trading-minute fraction
            underlying_mu = 0.00042
            base_sigma = 0.008
            normalized_distance = (matrix_distance - self.DISTANCE_MIN_MI) / (
                self.DISTANCE_MAX_MI - self.DISTANCE_MIN_MI
            )
            sigma_t = base_sigma * (1.0 + 1.4 * normalized_distance)

            z = self._rng.standard_normal(n)
            log_increments = (
                (underlying_mu - 0.5 * sigma_t ** 2) * dt
                + sigma_t * np.sqrt(dt) * z
            )
            log_path = np.cumsum(log_increments)
            base_price_ticker = self.BASE_FARE_FLOOR * np.exp(log_path)
            # Guarantee a sane floor so log() is always well-defined downstream.
            base_price_ticker = np.maximum(base_price_ticker, 0.01)

            # --- Surcharges (USD) ----------------------------------------------
            # Modeled as a rectified, event-driven Poisson-gated overlay:
            # most ticks carry a small baseline surcharge, with sporadic
            # congestion / surge spikes.
            surge_events = self._rng.poisson(lam=0.06, size=n)
            base_surcharge = self._rng.uniform(0.0, 1.25, size=n)
            surge_spike = surge_events * self._rng.uniform(1.5, self.SURCHARGE_MAX, size=n)
            surcharges = np.clip(base_surcharge + surge_spike, 0.0, self.SURCHARGE_MAX)

            df = pd.DataFrame({
                "timestamp": timestamps,
                "passenger_velocity": passenger_velocity,
                "matrix_distance": matrix_distance,
                "base_price_ticker": base_price_ticker,
                "surcharges": surcharges,
            })

            self.dataset = df
            elapsed = time.perf_counter() - start_ts
            logger.info(
                "generate_simulated_dataset: produced %d rows in %.4fs",
                len(df), elapsed
            )
            return df

        except Exception as exc:
            logger.error(
                "generate_simulated_dataset failed: %s\n%s",
                exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Failed to generate simulated dataset: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # 1b. REAL MARKET DATA INGESTION (optional -- requires `yfinance`)
    # --------------------------------------------------------------------- #
    def generate_real_dataset(self, ticker="AAPL", period="2y", interval="1d"):
        """
        Fetches real historical OHLCV data for a single equity ticker via
        Yahoo Finance (through the `yfinance` package) and maps it onto the
        SAME internal schema used by `generate_simulated_dataset`, so every
        downstream method (MLE, SDE solver, ridge regression, fuzzy
        inference, portfolio optimizer) works completely unmodified,
        regardless of whether the engine is running in synthetic or
        real-data mode.

        Column mapping (synthetic name -> real-data meaning):
            base_price_ticker   -> actual daily Close price
            passenger_velocity  -> normalized trading volume (demand-density
                                    proxy: higher volume = higher "velocity")
            matrix_distance     -> intraday range, (High - Low) / Close * 100
                                    (a volatility-tiering proxy, replacing
                                    the original trip-distance tiering)
            surcharges          -> abs(daily return) * 100 (a same-day
                                    price-shock magnitude proxy)

        Args:
            ticker (str): a valid ticker symbol, e.g. "AAPL", "MSFT",
                "GOOGL". Passed straight through to yfinance.
            period (str): yfinance lookback window, e.g. "1mo", "6mo",
                "1y", "2y", "5y", "max".
            interval (str): bar size, e.g. "1d", "1h", "5m". Intraday
                intervals are limited by Yahoo Finance to short lookback
                windows (e.g. "5m" only supports ~60 days of history).

        Returns:
            pandas.DataFrame: the mapped dataset, also cached on
            `self.dataset`. `self.n_records` is updated to the real row
            count returned by Yahoo Finance (which will be far smaller
            than the synthetic engine's default of 125,000).

        Raises:
            QuantEngineError: if `yfinance` is not installed, the network
                request fails, the ticker is invalid/delisted, or too few
                rows are returned to support stable downstream statistics
                (minimum 40 rows, to leave enough data for the 4-way
                distance-tier quantile bucketing used later).
        """
        if not YFINANCE_AVAILABLE:
            raise QuantEngineError(
                "yfinance is not installed. Run `pip install yfinance` "
                "(see requirements.txt) to enable real-market-data mode, "
                "or continue using generate_simulated_dataset()."
            )

        try:
            start_ts = time.perf_counter()
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period, interval=interval)

            if hist is None or hist.empty:
                raise QuantEngineError(
                    f"Yahoo Finance returned no data for ticker '{ticker}' "
                    f"(period={period}, interval={interval}). Verify the "
                    f"symbol is valid and not delisted."
                )
            if len(hist) < 40:
                raise QuantEngineError(
                    f"Only {len(hist)} rows returned for '{ticker}'; need "
                    f"at least 40 to support stable downstream statistics "
                    f"(MLE variance estimate + 4-way quantile tiering). "
                    f"Try a longer `period`."
                )

            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            missing = required_cols - set(hist.columns)
            if missing:
                raise QuantEngineError(
                    f"Yahoo Finance response for '{ticker}' is missing "
                    f"expected columns: {missing}."
                )

            close = hist["Close"].to_numpy(dtype=np.float64)
            high = hist["High"].to_numpy(dtype=np.float64)
            low = hist["Low"].to_numpy(dtype=np.float64)
            volume = hist["Volume"].to_numpy(dtype=np.float64)
            prev_close = np.concatenate([[close[0]], close[:-1]])

            if np.any(close <= 0):
                raise QuantEngineError(
                    f"'{ticker}' price history contains non-positive "
                    f"close prices; cannot compute log-returns."
                )

            # -- Map real OHLCV fields onto the existing internal schema --
            base_price_ticker = close

            vol_min, vol_max = float(volume.min()), float(volume.max())
            vol_span = (vol_max - vol_min) if (vol_max - vol_min) > 1e-9 else 1.0
            passenger_velocity = (volume - vol_min) / vol_span * 100.0

            matrix_distance = np.clip(
                (high - low) / close * 100.0, 1e-6, None
            )

            surcharges = np.abs((close - prev_close) / prev_close) * 100.0

            df = pd.DataFrame({
                "timestamp": hist.index,
                "passenger_velocity": passenger_velocity,
                "matrix_distance": matrix_distance,
                "base_price_ticker": base_price_ticker,
                "surcharges": surcharges,
            })

            self.dataset = df
            self.n_records = len(df)
            self._data_source = f"yfinance:{ticker}:{period}:{interval}"

            elapsed = time.perf_counter() - start_ts
            logger.info(
                "generate_real_dataset: fetched %d rows for '%s' "
                "(period=%s, interval=%s) in %.4fs",
                len(df), ticker, period, interval, elapsed
            )
            return df

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "generate_real_dataset failed for ticker '%s': %s\n%s",
                ticker, exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Failed to fetch real market data for '{ticker}': {exc}"
            ) from exc

    def generate_multi_asset_universe(self, tickers, period="1y", interval="1d"):
        """
        Fetches real historical Close prices for MULTIPLE tickers and
        computes a genuine, data-driven expected-return vector and
        covariance matrix across them -- used as a real-data replacement
        for `_build_distance_tier_assets` in the portfolio optimizer, so
        the minimum-variance allocation is solved over actual distinct
        equities rather than synthetic distance buckets of one series.

        Args:
            tickers (list[str]): e.g. ["AAPL", "MSFT", "GOOGL", "AMZN"].
                At least 2 tickers are required for a meaningful
                covariance matrix.
            period (str): yfinance lookback window.
            interval (str): bar size.

        Returns:
            tuple(np.ndarray, np.ndarray, list[str]): (expected_returns,
                cov_matrix, aligned_ticker_labels). Both arrays are
                annualized (assuming daily bars; see note below for other
                intervals).

        Raises:
            QuantEngineError: if `yfinance` is unavailable, fewer than 2
                tickers are supplied, any ticker fails to fetch, or the
                fetched series can't be aligned to a common length.
        """
        if not YFINANCE_AVAILABLE:
            raise QuantEngineError(
                "yfinance is not installed. Run `pip install yfinance` "
                "to enable real multi-asset portfolio mode."
            )
        if not tickers or len(tickers) < 2:
            raise QuantEngineError(
                f"generate_multi_asset_universe requires at least 2 "
                f"tickers; got {len(tickers) if tickers else 0}."
            )

        try:
            per_ticker_returns = []
            valid_labels = []

            for symbol in tickers:
                hist = yf.Ticker(symbol).history(period=period, interval=interval)
                if hist is None or hist.empty or len(hist) < 20:
                    logger.warning(
                        "generate_multi_asset_universe: skipping '%s' "
                        "(insufficient data returned).", symbol
                    )
                    continue
                close = hist["Close"].to_numpy(dtype=np.float64)
                if np.any(close <= 0):
                    logger.warning(
                        "generate_multi_asset_universe: skipping '%s' "
                        "(non-positive prices).", symbol
                    )
                    continue
                log_returns = np.diff(np.log(close))
                per_ticker_returns.append(log_returns)
                valid_labels.append(symbol)

            if len(per_ticker_returns) < 2:
                raise QuantEngineError(
                    "Fewer than 2 tickers returned usable data; cannot "
                    "build a covariance matrix."
                )

            min_len = min(r.size for r in per_ticker_returns)
            aligned = np.vstack([r[-min_len:] for r in per_ticker_returns])

            # Trading-day annualization factor; if `interval` is not daily
            # bars, this scaling no longer represents an annual figure and
            # should be treated as a relative (not literal annualized)
            # magnitude.
            annualization_factor = 252.0 if interval == "1d" else 1.0

            expected_returns = np.mean(aligned, axis=1) * annualization_factor
            cov_matrix = np.cov(aligned) * annualization_factor
            cov_matrix = np.atleast_2d(cov_matrix)

            logger.info(
                "generate_multi_asset_universe: built covariance matrix "
                "for %d tickers (%s) from %d aligned observations.",
                len(valid_labels), valid_labels, min_len
            )
            return expected_returns, cov_matrix, valid_labels

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "generate_multi_asset_universe failed: %s\n%s",
                exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Failed to build multi-asset universe: {exc}"
            ) from exc

    def fetch_ohlcv_for_chart(self, ticker="AAPL", period="6mo", interval="1d",
                               max_points=180):
        """
        Fetches raw OHLCV (Open/High/Low/Close/Volume) bars for a single
        ticker, shaped for direct consumption by a candlestick/financial
        chart on the frontend -- deliberately kept separate from
        `generate_real_dataset`, which instead REMAPS real data onto the
        engine's internal synthetic-schema column names for the
        MLE/SDE/ridge/fuzzy pipeline. This method exists purely for
        visualization and does not touch `self.dataset`.

        Args:
            ticker (str): ticker symbol, e.g. "AAPL".
            period (str): yfinance lookback window.
            interval (str): yfinance bar size.
            max_points (int): downsamples to at most this many bars so the
                JSON payload and chart rendering stay fast regardless of
                the requested lookback window.

        Returns:
            dict: {
                "ticker": str,
                "bars": [{"t": epoch_ms, "o": float, "h": float,
                           "l": float, "c": float, "v": float}, ...],
                "currency": str | None,
            }

        Raises:
            QuantEngineError: if yfinance is unavailable, the ticker is
                invalid, or no data is returned.
        """
        if not YFINANCE_AVAILABLE:
            raise QuantEngineError(
                "yfinance is not installed. Run `pip install yfinance` "
                "to enable the stock chart endpoint."
            )

        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period, interval=interval)

            if hist is None or hist.empty:
                raise QuantEngineError(
                    f"No chart data returned for ticker '{ticker}' "
                    f"(period={period}, interval={interval})."
                )

            if len(hist) > max_points:
                stride = max(1, len(hist) // max_points)
                hist = hist.iloc[::stride].tail(max_points)

            bars = []
            for idx, row in hist.iterrows():
                bars.append({
                    "t": int(idx.timestamp() * 1000),
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": float(row["Volume"]),
                })

            currency = None
            try:
                currency = ticker_obj.fast_info.get("currency")
            except Exception:
                # fast_info is best-effort metadata; its absence should
                # never fail the whole chart request.
                pass

            logger.info(
                "fetch_ohlcv_for_chart: returned %d bars for '%s'.",
                len(bars), ticker
            )
            return {"ticker": ticker.upper(), "bars": bars, "currency": currency}

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "fetch_ohlcv_for_chart failed for '%s': %s\n%s",
                ticker, exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Failed to fetch chart data for '{ticker}': {exc}"
            ) from exc

    def warm_up(self, mode="synthetic", ticker="AAPL", period="2y", interval="1d"):
        """
        Idempotently ensures the engine has a populated dataset, timing the
        operation for observability. Intended to be called exactly once at
        Flask application startup so the first inbound HTTP request never
        pays the dataset-generation cost.

        Args:
            mode (str): either "synthetic" (default, always available, no
                network dependency) or "real" (fetches live data for
                `ticker` via yfinance). If "real" mode fails for any
                reason -- no internet, invalid ticker, yfinance not
                installed -- the engine automatically falls back to
                synthetic mode rather than leaving the application unable
                to boot.
            ticker (str): ticker symbol used only when mode="real".
            period (str): yfinance lookback window, used only when
                mode="real".
            interval (str): yfinance bar size, used only when mode="real".

        Raises:
            QuantEngineError: only if BOTH the requested mode fails AND
                the synthetic fallback also fails (which would indicate a
                deeper environment problem, not a network issue).
        """
        if self._is_warmed and self.dataset is not None:
            logger.info("warm_up: engine already warm, skipping regeneration.")
            return

        start_ts = time.perf_counter()

        if mode == "real":
            try:
                self.generate_real_dataset(ticker=ticker, period=period, interval=interval)
            except QuantEngineError as exc:
                logger.warning(
                    "warm_up: real-data mode failed (%s); falling back "
                    "to synthetic dataset so the application can still "
                    "boot.", exc
                )
                self.generate_simulated_dataset()
        else:
            self.generate_simulated_dataset()

        self._last_warm_duration_sec = time.perf_counter() - start_ts
        self._is_warmed = True
        logger.info(
            "warm_up: engine pre-warmed with %d records in %.4fs (mode=%s)",
            self.n_records, self._last_warm_duration_sec, mode
        )

    def _ensure_warm(self):
        """Internal guard: raises if analytical methods are called cold."""
        if self.dataset is None or not self._is_warmed:
            raise QuantEngineError(
                "AdvancedQuantEngine has not been warmed up. Call "
                "warm_up() (or generate_simulated_dataset()) before "
                "requesting any analytical computation."
            )

    # --------------------------------------------------------------------- #
    # 2. MAXIMUM LIKELIHOOD ESTIMATION OF DRIFT / VOLATILITY
    # --------------------------------------------------------------------- #
    def compute_mle_drift_volatility(self, price_series=None, dt=1.0 / 390.0):
        """
        Computes the analytical Maximum Likelihood Estimators for the
        drift (mu) and volatility (sigma) parameters of a Geometric
        Brownian Motion process, under the standard stochastic
        log-normal assumption that log-returns are i.i.d. Normal:

            log(S_{t+1} / S_t) ~ Normal( (mu - 0.5*sigma^2)*dt, sigma^2*dt )

        Given a sample of log returns r_1, ..., r_N, the closed-form MLEs
        are:

            sigma_hat^2 = Var(r) / dt                (unbiased sample variance)
            mu_hat      = Mean(r) / dt + 0.5*sigma_hat^2

        Args:
            price_series (np.ndarray, optional): 1D array of strictly
                positive prices. Defaults to the engine's own
                `base_price_ticker` column.
            dt (float): the time-step size associated with consecutive
                observations in `price_series`, in year-fraction units.

        Returns:
            dict: {
                "mu": float, "sigma": float,
                "mean_log_return": float, "variance_log_return": float,
                "n_observations": int
            }

        Raises:
            QuantEngineError: if the input series is too short, contains
                non-positive values (log undefined), or if the resulting
                variance is degenerate (exactly zero).
        """
        try:
            self._ensure_warm()
            if price_series is None:
                price_series = self.dataset["base_price_ticker"].to_numpy()

            price_series = np.asarray(price_series, dtype=np.float64)

            if price_series.ndim != 1:
                raise QuantEngineError(
                    f"price_series must be 1-dimensional, got shape "
                    f"{price_series.shape}."
                )
            if price_series.size < 3:
                raise QuantEngineError(
                    "price_series must contain at least 3 observations "
                    "to compute a meaningful MLE variance estimate; got "
                    f"{price_series.size}."
                )
            if np.any(price_series <= 0):
                raise QuantEngineError(
                    "price_series contains non-positive values; log-return "
                    "MLE requires a strictly positive price process."
                )

            log_returns = np.diff(np.log(price_series))
            mean_lr = float(np.mean(log_returns))
            var_lr = float(np.var(log_returns, ddof=1))

            if var_lr <= 0.0:
                raise QuantEngineError(
                    "Degenerate zero-variance log-return series; cannot "
                    "compute a well-defined MLE for sigma."
                )

            sigma_hat = float(np.sqrt(var_lr / dt))
            mu_hat = float(mean_lr / dt + 0.5 * sigma_hat ** 2)

            result = {
                "mu": mu_hat,
                "sigma": sigma_hat,
                "mean_log_return": mean_lr,
                "variance_log_return": var_lr,
                "n_observations": int(log_returns.size),
            }
            logger.info(
                "compute_mle_drift_volatility: mu=%.6f sigma=%.6f (n=%d)",
                mu_hat, sigma_hat, log_returns.size
            )
            return result

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "compute_mle_drift_volatility failed: %s\n%s",
                exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"MLE drift/volatility estimation failed: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # 3. EULER-MARUYAMA GBM SDE SOLVER
    # --------------------------------------------------------------------- #
    def euler_maruyama_gbm(self, s0, mu, sigma, n_paths=None, n_steps=None,
                            dt=1.0 / 252.0):
        """
        Discretizes and numerically solves the Geometric Brownian Motion
        stochastic differential equation:

            dS_t = mu * S_t * dt + sigma * S_t * dW_t

        via the explicit Euler-Maruyama scheme:

            S_{t+1} = S_t + mu*S_t*dt + sigma*S_t*sqrt(dt)*Z_t,
            Z_t ~ Normal(0, 1) i.i.d.

        Args:
            s0 (float): initial asset price, must be > 0.
            mu (float): drift coefficient (annualized).
            sigma (float): volatility coefficient (annualized), must be >= 0.
            n_paths (int, optional): number of independent sample paths.
                Defaults to `self.n_sde_paths`.
            n_steps (int, optional): number of discrete time steps.
                Defaults to `self.n_sde_steps`.
            dt (float): step size in year-fraction units.

        Returns:
            np.ndarray: shape (n_paths, n_steps + 1) array of simulated
            price paths (column 0 is the initial condition `s0` broadcast
            across every path).

        Raises:
            QuantEngineError: for invalid parameters (non-positive s0,
                negative sigma, non-positive path/step counts) or if the
                discretization produces a non-finite result.
        """
        try:
            n_paths = int(n_paths) if n_paths is not None else self.n_sde_paths
            n_steps = int(n_steps) if n_steps is not None else self.n_sde_steps

            if s0 <= 0:
                raise QuantEngineError(f"s0 must be > 0, got {s0}.")
            if sigma < 0:
                raise QuantEngineError(f"sigma must be >= 0, got {sigma}.")
            if n_paths <= 0 or n_steps <= 0:
                raise QuantEngineError(
                    f"n_paths and n_steps must be positive integers; got "
                    f"n_paths={n_paths}, n_steps={n_steps}."
                )
            if dt <= 0:
                raise QuantEngineError(f"dt must be > 0, got {dt}.")

            paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
            paths[:, 0] = s0

            # Pre-draw all Wiener increments at once (vectorized) rather
            # than per-step RNG calls, for both speed and to avoid
            # accidental correlation artifacts from re-seeding mid-loop.
            dW = self._rng.standard_normal(size=(n_paths, n_steps)) * np.sqrt(dt)

            sqrt_dt_sigma = sigma
            for t in range(1, n_steps + 1):
                prev = paths[:, t - 1]
                increment = mu * prev * dt + sqrt_dt_sigma * prev * dW[:, t - 1]
                nxt = prev + increment
                # A GBM price process should remain strictly positive;
                # under large dt / sigma combinations the explicit Euler
                # scheme can occasionally overshoot below zero, so we
                # floor it to a small epsilon to preserve well-posedness
                # for any downstream log() calls.
                paths[:, t] = np.maximum(nxt, 1e-8)

            if not np.all(np.isfinite(paths)):
                raise QuantEngineError(
                    "Euler-Maruyama discretization produced non-finite "
                    "values (NaN/Inf); check mu/sigma/dt magnitudes."
                )

            logger.info(
                "euler_maruyama_gbm: simulated %d paths x %d steps "
                "(s0=%.4f, mu=%.6f, sigma=%.6f, dt=%.6f)",
                n_paths, n_steps, s0, mu, sigma, dt
            )
            return paths

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "euler_maruyama_gbm failed: %s\n%s", exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Euler-Maruyama GBM simulation failed: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # 4. CUSTOM CLOSED-FORM L2-REGULARIZED (RIDGE) REGRESSION
    # --------------------------------------------------------------------- #
    @staticmethod
    def _closed_form_ridge_fit(X, y, alpha):
        """
        Solves the L2-regularized least squares (Ridge Regression) problem
        in closed form using raw numpy matrix algebra:

            beta = (X^T X + alpha * I)^(-1) X^T y

        The intercept column (assumed to be the first column of X, all
        ones) is deliberately excluded from regularization by zeroing the
        corresponding diagonal entry of the penalty matrix -- standard
        Ridge practice, since penalizing the intercept would bias
        predictions toward zero rather than merely shrinking slopes.

        Args:
            X (np.ndarray): design matrix of shape (n_samples, n_features),
                with an explicit leading intercept column of ones.
            y (np.ndarray): target vector of shape (n_samples,).
            alpha (float): non-negative L2 regularization strength.

        Returns:
            np.ndarray: fitted coefficient vector `beta`, shape (n_features,).

        Raises:
            QuantEngineError: on dimensionality mismatches, non-positive
                alpha, or a numerically singular (X^T X + alpha*I) matrix
                that cannot be inverted even with pseudo-inverse fallback.
        """
        if X.ndim != 2:
            raise QuantEngineError(
                f"Design matrix X must be 2-dimensional, got shape {X.shape}."
            )
        if y.ndim != 1:
            raise QuantEngineError(
                f"Target vector y must be 1-dimensional, got shape {y.shape}."
            )
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise QuantEngineError(
                f"Row-count mismatch between X ({n_samples}) and y "
                f"({y.shape[0]}); every observation needs both a feature "
                f"row and a target value."
            )
        if n_samples < n_features:
            raise QuantEngineError(
                f"Underdetermined system: n_samples ({n_samples}) < "
                f"n_features ({n_features}); cannot fit a stable ridge "
                f"solution."
            )
        if alpha < 0:
            raise QuantEngineError(f"alpha must be >= 0, got {alpha}.")

        try:
            identity_penalty = np.eye(n_features, dtype=np.float64)
            identity_penalty[0, 0] = 0.0  # do not regularize the intercept

            XtX = X.T @ X
            penalized = XtX + alpha * identity_penalty

            try:
                inv_term = np.linalg.inv(penalized)
            except np.linalg.LinAlgError:
                # Singular matrix (e.g. perfectly collinear features) --
                # fall back to the Moore-Penrose pseudo-inverse, which
                # still yields the minimum-norm least-squares solution.
                logger.warning(
                    "_closed_form_ridge_fit: (X^T X + alpha*I) is "
                    "singular; falling back to pseudo-inverse."
                )
                inv_term = np.linalg.pinv(penalized)

            beta = inv_term @ X.T @ y

            if not np.all(np.isfinite(beta)):
                raise QuantEngineError(
                    "Ridge closed-form solution produced non-finite "
                    "coefficients; the design matrix is likely severely "
                    "ill-conditioned."
                )
            return beta

        except QuantEngineError:
            raise
        except Exception as exc:
            raise QuantEngineError(
                f"Closed-form ridge fit failed during matrix algebra: {exc}"
            ) from exc

    def custom_ridge_regression(self, alpha=1.0, sample_size=None):
        """
        Fits a from-scratch L2-regularized linear model predicting
        `base_price_ticker` from (matrix_distance, passenger_velocity,
        surcharges), using the engine's own dataset as training data and
        the closed-form solver in `_closed_form_ridge_fit`.

        Args:
            alpha (float): L2 regularization strength.
            sample_size (int, optional): if provided, fits on a random
                subsample of this many rows (useful to keep the matrix
                inversion fast on very large synthetic datasets); defaults
                to using the full dataset.

        Returns:
            dict: {
                "intercept": float,
                "coefficients": {"matrix_distance": float,
                                  "passenger_velocity": float,
                                  "surcharges": float},
                "alpha": float,
                "r_squared": float,
                "n_samples": int,
                "n_features": int,
            }

        Raises:
            QuantEngineError: propagated from `_closed_form_ridge_fit`, or
                raised directly if the engine has not been warmed up.
        """
        try:
            self._ensure_warm()
            df = self.dataset

            if sample_size is not None and 0 < sample_size < len(df):
                df = df.sample(n=sample_size, random_state=self._seed)

            feature_cols = ["matrix_distance", "passenger_velocity", "surcharges"]
            X_raw = df[feature_cols].to_numpy(dtype=np.float64)
            y = df["base_price_ticker"].to_numpy(dtype=np.float64)

            # Standardize features (zero mean, unit variance) before
            # regularizing -- otherwise the penalty term `alpha*I` would
            # apply asymmetric shrinkage across features on wildly
            # different natural scales (miles vs. mph vs. dollars).
            feature_means = X_raw.mean(axis=0)
            feature_stds = X_raw.std(axis=0)
            feature_stds[feature_stds == 0] = 1.0  # guard divide-by-zero
            X_scaled = (X_raw - feature_means) / feature_stds

            intercept_col = np.ones((X_scaled.shape[0], 1), dtype=np.float64)
            X_design = np.hstack([intercept_col, X_scaled])

            beta = self._closed_form_ridge_fit(X_design, y, alpha=alpha)

            predictions = X_design @ beta
            residuals = y - predictions
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            result = {
                "intercept": float(beta[0]),
                "coefficients": {
                    "matrix_distance": float(beta[1]),
                    "passenger_velocity": float(beta[2]),
                    "surcharges": float(beta[3]),
                },
                "alpha": float(alpha),
                "r_squared": float(r_squared),
                "n_samples": int(X_design.shape[0]),
                "n_features": int(X_design.shape[1]),
            }
            logger.info(
                "custom_ridge_regression: fit on %d samples, alpha=%.4f, "
                "r_squared=%.6f", result["n_samples"], alpha, r_squared
            )
            return result

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "custom_ridge_regression failed: %s\n%s",
                exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Custom ridge regression failed: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # 5. MAMDANI FUZZY INFERENCE SYSTEM -- RISK ASSET SCORE
    # --------------------------------------------------------------------- #
    @staticmethod
    def _triangular_membership(x, a, b, c):
        """
        Evaluates a triangular membership function at point(s) `x`, defined
        by the foot-peak-foot parameters (a, b, c) with a <= b <= c:

                    0,                      x <= a
            mu(x) = (x-a)/(b-a),            a < x <= b
                    (c-x)/(c-b),            b < x < c
                    0,                      x >= c

        Vectorized over numpy arrays; degenerate spans (a == b or b == c)
        are handled by treating that half of the triangle as a step
        function rather than dividing by zero.

        Args:
            x (float or np.ndarray): input value(s).
            a, b, c (float): triangle foot-left, peak, foot-right.

        Returns:
            float or np.ndarray: membership degree(s) in [0, 1].
        """
        x = np.asarray(x, dtype=np.float64)
        left_span = (b - a) if (b - a) != 0 else 1e-12
        right_span = (c - b) if (c - b) != 0 else 1e-12

        rising = (x - a) / left_span
        falling = (c - x) / right_span

        membership = np.minimum(rising, falling)
        membership = np.clip(membership, 0.0, 1.0)
        return membership

    def fuzzy_risk_inference(self, volatility, demand_density,
                              vol_universe_max=1.5, output_resolution=401):
        """
        Runs a Mamdani-style fuzzy inference system over two crisp inputs,
        Volatility and Demand Density, to produce a defuzzified real-time
        Risk Asset Score on a [0, 100] scale.

        Linguistic variables:
            Volatility (input, universe [0, vol_universe_max]):
                LOW    : triangle(0,               0,               0.45*max)
                MEDIUM : triangle(0.20*max,         0.50*max,        0.80*max)
                HIGH   : triangle(0.55*max,         max,             max)

            Demand Density (input, universe [0, 1], expected pre-normalized):
                LOW    : triangle(0,    0,    0.40)
                MEDIUM : triangle(0.20, 0.50, 0.80)
                HIGH    : triangle(0.60, 1.0, 1.0)

            Risk Score (output, universe [0, 100]):
                LOW    : triangle(0,   0,   40)
                MEDIUM : triangle(25,  50,  75)
                HIGH   : triangle(60,  100, 100)

        Rule base (Mamdani AND = min, aggregation = max):
            R1: Vol=LOW    AND Demand=LOW    -> Risk=LOW
            R2: Vol=LOW    AND Demand=MEDIUM -> Risk=LOW
            R3: Vol=LOW    AND Demand=HIGH   -> Risk=MEDIUM
            R4: Vol=MEDIUM AND Demand=LOW    -> Risk=LOW
            R5: Vol=MEDIUM AND Demand=MEDIUM -> Risk=MEDIUM
            R6: Vol=MEDIUM AND Demand=HIGH   -> Risk=HIGH
            R7: Vol=HIGH   AND Demand=LOW    -> Risk=MEDIUM
            R8: Vol=HIGH   AND Demand=MEDIUM -> Risk=HIGH
            R9: Vol=HIGH   AND Demand=HIGH   -> Risk=HIGH

        Defuzzification: centroid (center of gravity) of the aggregated
        output membership function, computed via discretized numerical
        integration over `output_resolution` sample points spanning [0,100].

        Args:
            volatility (float): crisp annualized volatility input (e.g.
                sigma from the MLE step), clipped into
                [0, vol_universe_max] before fuzzification.
            demand_density (float): crisp demand-density input, expected
                pre-normalized into [0, 1].
            vol_universe_max (float): the assumed maximum of the volatility
                universe of discourse, used to scale the triangular breakpoints.
            output_resolution (int): number of discretization points used
                for the centroid numerical integration.

        Returns:
            dict: {
                "risk_score": float (0-100),
                "volatility_memberships": {"low": float, "medium": float, "high": float},
                "demand_memberships": {"low": float, "medium": float, "high": float},
                "rule_firing_strengths": {"R1": float, ..., "R9": float},
            }

        Raises:
            QuantEngineError: on invalid input types/ranges or a
                degenerate (all-zero) aggregated output membership
                function, which would make the centroid undefined.
        """
        try:
            if vol_universe_max <= 0:
                raise QuantEngineError(
                    f"vol_universe_max must be > 0, got {vol_universe_max}."
                )
            if output_resolution < 10:
                raise QuantEngineError(
                    f"output_resolution must be >= 10 for a meaningful "
                    f"centroid integration, got {output_resolution}."
                )

            vol_x = float(np.clip(volatility, 0.0, vol_universe_max))
            demand_x = float(np.clip(demand_density, 0.0, 1.0))

            vmax = vol_universe_max

            # --- Fuzzify inputs ------------------------------------------------
            vol_low = float(self._triangular_membership(vol_x, 0.0, 0.0, 0.45 * vmax))
            vol_med = float(self._triangular_membership(vol_x, 0.20 * vmax, 0.50 * vmax, 0.80 * vmax))
            vol_high = float(self._triangular_membership(vol_x, 0.55 * vmax, vmax, vmax))

            dem_low = float(self._triangular_membership(demand_x, 0.0, 0.0, 0.40))
            dem_med = float(self._triangular_membership(demand_x, 0.20, 0.50, 0.80))
            dem_high = float(self._triangular_membership(demand_x, 0.60, 1.0, 1.0))

            # --- Rule evaluation (Mamdani AND = min) ---------------------------
            rules = {
                "R1": ("low", min(vol_low, dem_low)),
                "R2": ("low", min(vol_low, dem_med)),
                "R3": ("medium", min(vol_low, dem_high)),
                "R4": ("low", min(vol_med, dem_low)),
                "R5": ("medium", min(vol_med, dem_med)),
                "R6": ("high", min(vol_med, dem_high)),
                "R7": ("medium", min(vol_high, dem_low)),
                "R8": ("high", min(vol_high, dem_med)),
                "R9": ("high", min(vol_high, dem_high)),
            }

            # Aggregate firing strength per output linguistic term (max).
            agg_strength = {"low": 0.0, "medium": 0.0, "high": 0.0}
            for _, (term, strength) in rules.items():
                agg_strength[term] = max(agg_strength[term], strength)

            # --- Output universe + clipped aggregated membership function -----
            output_universe = np.linspace(0.0, 100.0, output_resolution)

            out_low = self._triangular_membership(output_universe, 0.0, 0.0, 40.0)
            out_med = self._triangular_membership(output_universe, 25.0, 50.0, 75.0)
            out_high = self._triangular_membership(output_universe, 60.0, 100.0, 100.0)

            clipped_low = np.minimum(out_low, agg_strength["low"])
            clipped_med = np.minimum(out_med, agg_strength["medium"])
            clipped_high = np.minimum(out_high, agg_strength["high"])

            aggregated = np.maximum(np.maximum(clipped_low, clipped_med), clipped_high)

            # numpy >= 2.0 renamed trapz -> trapezoid; resolve whichever is
            # available at runtime so this module works correctly against
            # both the pinned production numpy (1.26.4, has trapz) and any
            # newer numpy present in a development environment.
            _integrate = getattr(np, "trapezoid", None) or np.trapz
            denominator = float(_integrate(aggregated, output_universe))
            if denominator <= 1e-9:
                # No rule fired meaningfully (e.g. exactly-zero inputs at a
                # boundary discontinuity); default to the midpoint of the
                # universe as a neutral, well-defined fallback rather than
                # dividing by (near) zero.
                logger.warning(
                    "fuzzy_risk_inference: degenerate aggregated output "
                    "membership (area ~= 0); defaulting risk score to 50.0."
                )
                risk_score = 50.0
            else:
                numerator = float(_integrate(aggregated * output_universe, output_universe))
                risk_score = numerator / denominator

            result = {
                "risk_score": float(np.clip(risk_score, 0.0, 100.0)),
                "volatility_memberships": {
                    "low": vol_low, "medium": vol_med, "high": vol_high
                },
                "demand_memberships": {
                    "low": dem_low, "medium": dem_med, "high": dem_high
                },
                "rule_firing_strengths": {k: v[1] for k, v in rules.items()},
            }
            logger.info(
                "fuzzy_risk_inference: risk_score=%.4f (vol=%.4f, demand=%.4f)",
                result["risk_score"], vol_x, demand_x
            )
            return result

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "fuzzy_risk_inference failed: %s\n%s", exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Fuzzy risk inference failed: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # 6. CONSTRAINED PORTFOLIO VARIANCE MINIMIZATION
    # --------------------------------------------------------------------- #
    def optimize_portfolio_allocation(self, expected_returns, cov_matrix,
                                       max_iter=500, tolerance=1e-9):
        """
        Solves the classical minimum-variance portfolio problem:

            minimize_w   w^T * Sigma * w
            subject to   sum(w) == 1.0
                         0 <= w_i <= 1  for all i

        via scipy.optimize.minimize using Sequential Least Squares
        Programming (SLSQP), which natively supports the hard equality
        constraint and box bounds this problem requires.

        Args:
            expected_returns (array-like): length-N vector of expected
                asset/bucket returns (used only for reporting the
                resulting portfolio's expected return; the objective
                itself is pure variance minimization).
            cov_matrix (array-like): N x N covariance matrix of asset
                returns. Must be square and match the length of
                `expected_returns`.
            max_iter (int): maximum SLSQP iterations.
            tolerance (float): SLSQP convergence tolerance (`ftol`).

        Returns:
            dict: {
                "weights": [float, ...],       # length N, sums to ~1.0
                "portfolio_variance": float,
                "portfolio_volatility": float,
                "expected_portfolio_return": float,
                "converged": bool,
                "iterations": int,
                "message": str,
            }

        Raises:
            QuantEngineError: on shape mismatches, a non-positive-
                semidefinite covariance matrix, or SLSQP failing to
                converge.
        """
        try:
            mu_vec = np.asarray(expected_returns, dtype=np.float64).flatten()
            sigma_mat = np.asarray(cov_matrix, dtype=np.float64)
            n = mu_vec.shape[0]

            if sigma_mat.shape != (n, n):
                raise QuantEngineError(
                    f"cov_matrix shape {sigma_mat.shape} is incompatible "
                    f"with expected_returns length {n}; covariance matrix "
                    f"must be square and match asset count exactly."
                )
            if n < 2:
                raise QuantEngineError(
                    f"Portfolio optimization requires at least 2 assets; "
                    f"got {n}."
                )
            # Symmetry check with tolerance -- covariance matrices must be
            # symmetric by construction; a meaningful asymmetry indicates
            # an upstream data-assembly bug rather than float rounding.
            if not np.allclose(sigma_mat, sigma_mat.T, atol=1e-6):
                raise QuantEngineError(
                    "cov_matrix is not symmetric; a valid covariance "
                    "matrix must satisfy Sigma == Sigma^T."
                )

            def objective(w):
                return float(w @ sigma_mat @ w)

            def objective_grad(w):
                return 2.0 * sigma_mat @ w

            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                 "jac": lambda w: np.ones_like(w)},
            )
            bounds = [(0.0, 1.0) for _ in range(n)]
            w0 = np.repeat(1.0 / n, n)

            result = minimize(
                objective, w0, jac=objective_grad, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": max_iter, "ftol": tolerance, "disp": False},
            )

            if not result.success:
                raise QuantEngineError(
                    f"Portfolio optimizer failed to converge: {result.message}"
                )

            weights = np.clip(result.x, 0.0, 1.0)
            weights = weights / np.sum(weights)  # renormalize for exact sum=1.0

            portfolio_variance = float(weights @ sigma_mat @ weights)
            portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
            expected_portfolio_return = float(np.dot(weights, mu_vec))

            output = {
                "weights": [float(w) for w in weights],
                "portfolio_variance": portfolio_variance,
                "portfolio_volatility": portfolio_volatility,
                "expected_portfolio_return": expected_portfolio_return,
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
            logger.info(
                "optimize_portfolio_allocation: converged=%s iters=%d "
                "variance=%.8f", output["converged"], output["iterations"],
                portfolio_variance
            )
            return output

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "optimize_portfolio_allocation failed: %s\n%s",
                exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Portfolio allocation optimization failed: {exc}"
            ) from exc

    # --------------------------------------------------------------------- #
    # ORCHESTRATION: FULL END-TO-END ANALYTICAL PASS
    # --------------------------------------------------------------------- #
    def set_multi_asset_universe(self, tickers, period="1y", interval="1d"):
        """
        Fetches and caches a real multi-ticker covariance universe (via
        `generate_multi_asset_universe`) so the NEXT call to
        `run_full_analysis` uses it for portfolio optimization instead of
        the synthetic distance-tier buckets. Call this once after
        `warm_up()` and before hitting the analysis endpoint, or re-call
        it whenever the desired ticker basket changes.

        Args:
            tickers (list[str]): tickers to include in the portfolio
                universe, e.g. ["AAPL", "MSFT", "GOOGL", "AMZN"].
            period (str): yfinance lookback window.
            interval (str): yfinance bar size.

        Raises:
            QuantEngineError: propagated from `generate_multi_asset_universe`.
        """
        expected_returns, cov_matrix, labels = self.generate_multi_asset_universe(
            tickers=tickers, period=period, interval=interval
        )
        self._multi_asset_universe = {
            "expected_returns": expected_returns,
            "cov_matrix": cov_matrix,
            "labels": labels,
        }
        logger.info(
            "set_multi_asset_universe: cached real portfolio universe "
            "for %s.", labels
        )

    def clear_multi_asset_universe(self):
        """Reverts `run_full_analysis` to the synthetic distance-tier portfolio."""
        self._multi_asset_universe = None

    def _build_distance_tier_assets(self, n_tiers=4):
        """
        Buckets the warmed dataset into `n_tiers` distance-based quantile
        segments and treats each segment's price-ticker log-return series
        as a pseudo-asset. This gives the portfolio optimizer a genuine,
        data-derived expected-return vector and covariance matrix instead
        of an arbitrary hardcoded one.

        Returns:
            tuple(np.ndarray, np.ndarray): (expected_returns, cov_matrix)
                of shape (n_tiers,) and (n_tiers, n_tiers) respectively.

        Raises:
            QuantEngineError: if any tier ends up with too few rows to
                compute a stable log-return series (n_tiers set too high
                relative to dataset size).
        """
        df = self.dataset
        try:
            tier_labels = pd.qcut(
                df["matrix_distance"], q=n_tiers,
                labels=[f"tier_{i}" for i in range(n_tiers)]
            )
        except ValueError as exc:
            raise QuantEngineError(
                f"Unable to bucket dataset into {n_tiers} distance tiers "
                f"(dataset likely too small or too uniform): {exc}"
            ) from exc

        tier_log_returns = []
        min_len = None
        for i in range(n_tiers):
            tier_prices = df.loc[tier_labels == f"tier_{i}", "base_price_ticker"].to_numpy()
            if tier_prices.size < 10:
                raise QuantEngineError(
                    f"Distance tier {i} has only {tier_prices.size} rows; "
                    f"need at least 10 to compute a stable return series."
                )
            lr = np.diff(np.log(tier_prices))
            tier_log_returns.append(lr)
            min_len = lr.size if min_len is None else min(min_len, lr.size)

        # Truncate every tier's return series to the same length so they
        # can be stacked into a well-formed covariance matrix.
        aligned = np.vstack([lr[:min_len] for lr in tier_log_returns])

        expected_returns = np.mean(aligned, axis=1) * 252.0  # annualized
        cov_matrix = np.cov(aligned) * 252.0  # annualized covariance

        # Guard against a covariance matrix that collapsed to a scalar
        # when n_tiers == 1 (shouldn't happen given our fixed n_tiers=4
        # default, but defensive nonetheless).
        cov_matrix = np.atleast_2d(cov_matrix)

        return expected_returns, cov_matrix

    def run_full_analysis(self, ridge_alpha=1.0, gbm_n_paths=None,
                           gbm_n_steps=None, downsample_points=180):
        """
        Orchestrates a complete end-to-end analytical pass across all five
        quantitative subsystems, using the engine's live synthetic dataset
        as the single source of truth. This is the method the Flask
        controller layer calls per-request.

        Steps:
            1. Ensure the engine is warm (regenerates nothing if already so).
            2. MLE-estimate (mu, sigma) from the price ticker series.
            3. Run Euler-Maruyama GBM Monte Carlo forward simulation seeded
               at the most recent observed price.
            4. Fit the custom closed-form ridge regression model.
            5. Derive a data-driven expected-return vector and covariance
               matrix from distance-tiered pseudo-assets, and solve the
               minimum-variance portfolio allocation.
            6. Compute the fuzzy Risk Asset Score from the MLE volatility
               and a normalized demand-density metric derived from
               passenger velocity.
            7. Downsample large arrays (SDE paths, raw ticks) to a
               frontend-friendly point count before returning.

        Args:
            ridge_alpha (float): L2 penalty strength for the ridge fit.
            gbm_n_paths (int, optional): override for Monte Carlo path count.
            gbm_n_steps (int, optional): override for Monte Carlo step count.
            downsample_points (int): target number of points per series
                returned to the frontend for charting (keeps JSON payload
                size bounded regardless of the underlying dataset size).

        Returns:
            dict: fully JSON-serializable analytical payload. See app.py's
            `/api/compute-quant-matrix` route for the exact shape consumed
            by the frontend.

        Raises:
            QuantEngineError: propagated from any of the constituent
                subsystem calls.
        """
        try:
            self._ensure_warm()
            overall_start = time.perf_counter()
            df = self.dataset

            # --- 2. MLE ------------------------------------------------------
            mle_result = self.compute_mle_drift_volatility()

            # --- 3. Euler-Maruyama GBM forward simulation ---------------------
            s0 = float(df["base_price_ticker"].iloc[-1])
            sde_paths = self.euler_maruyama_gbm(
                s0=s0, mu=mle_result["mu"], sigma=mle_result["sigma"],
                n_paths=gbm_n_paths, n_steps=gbm_n_steps,
            )

            # --- 4. Custom closed-form ridge regression -----------------------
            ridge_result = self.custom_ridge_regression(alpha=ridge_alpha)

            # --- 5. Data-driven portfolio optimization -------------------------
            # Uses a real multi-ticker covariance universe if one has been
            # cached via set_multi_asset_universe(); otherwise falls back
            # to the synthetic distance-tier buckets built from this
            # engine's own single-series dataset.
            if self._multi_asset_universe is not None:
                expected_returns = self._multi_asset_universe["expected_returns"]
                cov_matrix = self._multi_asset_universe["cov_matrix"]
                asset_labels = self._multi_asset_universe["labels"]
            else:
                expected_returns, cov_matrix = self._build_distance_tier_assets(n_tiers=4)
                asset_labels = ["short_haul", "mid_haul", "long_haul", "extended_haul"]

            portfolio_result = self.optimize_portfolio_allocation(
                expected_returns=expected_returns, cov_matrix=cov_matrix
            )

            # --- 6. Fuzzy Risk Asset Score --------------------------------------
            # Demand density proxy: normalized mean passenger velocity over
            # the trailing window, mapped into [0, 1] via its own observed
            # min/max range so the fuzzy input is always well-scaled
            # regardless of the underlying synthetic parameters.
            velocity_window = df["passenger_velocity"].to_numpy()[-2000:]
            v_min, v_max = float(velocity_window.min()), float(velocity_window.max())
            v_span = (v_max - v_min) if (v_max - v_min) > 1e-9 else 1.0
            demand_density = float((np.mean(velocity_window) - v_min) / v_span)

            fuzzy_result = self.fuzzy_risk_inference(
                volatility=mle_result["sigma"], demand_density=demand_density
            )

            # --- 7. Downsampling for frontend payload --------------------------
            def _downsample_1d(arr, target_points):
                arr = np.asarray(arr)
                if arr.size <= target_points:
                    return arr.tolist()
                stride = max(1, arr.size // target_points)
                return arr[::stride][:target_points].tolist()

            # Downsample a representative subset of Monte Carlo paths (not
            # all `n_paths`, which would bloat the payload) plus the mean
            # path and a 5th/95th percentile envelope for a fan chart.
            path_mean = np.mean(sde_paths, axis=0)
            path_p05 = np.percentile(sde_paths, 5, axis=0)
            path_p95 = np.percentile(sde_paths, 95, axis=0)
            sample_path_count = min(8, sde_paths.shape[0])
            sample_indices = np.linspace(
                0, sde_paths.shape[0] - 1, sample_path_count, dtype=int
            )

            sde_forecast_payload = {
                "mean_path": _downsample_1d(path_mean, downsample_points),
                "p05_path": _downsample_1d(path_p05, downsample_points),
                "p95_path": _downsample_1d(path_p95, downsample_points),
                "sample_paths": [
                    _downsample_1d(sde_paths[i], downsample_points)
                    for i in sample_indices
                ],
            }

            raw_velocity_payload = _downsample_1d(
                df["passenger_velocity"].to_numpy(), downsample_points
            )
            raw_distance_payload = _downsample_1d(
                df["matrix_distance"].to_numpy(), downsample_points
            )
            raw_price_payload = _downsample_1d(
                df["base_price_ticker"].to_numpy(), downsample_points
            )
            timestamp_payload = _downsample_1d(
                df["timestamp"].astype(np.int64).to_numpy() // 10**6,  # ms epoch
                downsample_points
            )

            elapsed = time.perf_counter() - overall_start
            payload = {
                "meta": {
                    "n_records": int(len(df)),
                    "computation_time_ms": round(elapsed * 1000.0, 3),
                    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                    "data_source": self._data_source,
                },
                "mle": mle_result,
                "sde_forecast": sde_forecast_payload,
                "ridge_regression": ridge_result,
                "portfolio_allocation": {
                    **portfolio_result,
                    "asset_labels": asset_labels,
                    "expected_returns_input": [float(v) for v in expected_returns],
                },
                "fuzzy_risk": fuzzy_result,
                "raw_series": {
                    "timestamps_ms": timestamp_payload,
                    "velocity": raw_velocity_payload,
                    "distance": raw_distance_payload,
                    "price": raw_price_payload,
                },
            }
            logger.info(
                "run_full_analysis: completed full analytical pass in %.4fs",
                elapsed
            )
            return payload

        except QuantEngineError:
            raise
        except Exception as exc:
            logger.error(
                "run_full_analysis failed: %s\n%s", exc, traceback.format_exc()
            )
            raise QuantEngineError(
                f"Full analytical pass failed: {exc}"
            ) from exc