/**
 * dashboard.js
 * ============================================================================
 * THE CLIENT OPERATIONS SCRIPT
 * ----------------------------------------------------------------------------
 * Native ES6, class-based client engine for the High-Frequency Macro
 * Algorithmic Arbitrage Engine terminal. Responsibilities:
 *
 *   - poll `GET /api/compute-quant-matrix` on a fixed interval to simulate
 *     an incoming high-frequency analytical data stream,
 *   - parse the returned JSON analytical vectors defensively,
 *   - drive three Chart.js canvases (SDE forecast fan chart, portfolio
 *     allocation pie, raw velocity ticker) plus a KPI strip, a fuzzy-risk
 *     gauge, a ridge-regression coefficient table, fuzzy membership bars,
 *     and a live system-status readout.
 *
 * No build step, no framework -- vanilla ES6 only, entirely bound to the
 * DOMContentLoaded lifecycle hook per the project's architectural brief.
 * ============================================================================
 */

(function () {
    "use strict";

    /** Endpoint the dashboard polls for the full analytical payload. */
    const COMPUTE_ENDPOINT = "/api/compute-quant-matrix";

    /** Polling cadence, in milliseconds, simulating a live tick stream. */
    const POLL_INTERVAL_MS = 4000;

    /** Hard network timeout for a single fetch, in milliseconds. */
    const FETCH_TIMEOUT_MS = 12000;

    /** Shared color tokens, kept in sync with style.css custom properties. */
    const COLORS = {
        cyan: "#35e6c6",
        cyanDim: "rgba(53, 230, 198, 0.14)",
        amber: "#ffb443",
        amberDim: "rgba(255, 180, 67, 0.14)",
        magenta: "#ff4d7e",
        violet: "#8b7cff",
        violetDim: "rgba(139, 124, 255, 0.35)",
        textPrimary: "#e8edf9",
        textSecondary: "#8493b4",
        textTertiary: "#56628a",
        gridLine: "rgba(36, 49, 79, 0.55)",
    };

    /**
     * QuantDashboard
     * --------------------------------------------------------------------
     * Encapsulates the entire client-side lifecycle: DOM element caching,
     * chart instantiation, network polling, and per-frame UI updates.
     */
    class QuantDashboard {
        constructor() {
            this.els = {};
            this.charts = {
                forecast: null,
                allocation: null,
                velocity: null,
            };
            this.requestCount = 0;
            this.isPolling = false;
            this.pollTimerId = null;
            this.assetPalette = [COLORS.cyan, COLORS.violet, COLORS.amber, COLORS.magenta];
        }

        /**
         * Entry point: caches DOM references, boots the live clock,
         * instantiates empty chart shells, then performs an immediate
         * fetch before handing off to the recurring polling loop.
         */
        init() {
            this._cacheDomRefs();
            this._startClock();
            this._initCharts();
            this._pollOnce();
            this._startPolling();

            window.addEventListener("beforeunload", () => this._stopPolling());
        }

        // -------------------------------------------------------------- //
        // DOM CACHING
        // -------------------------------------------------------------- //
        _cacheDomRefs() {
            const ids = [
                "connectionDot", "connectionLabel", "recordCountValue",
                "lastTickValue", "computeTimeValue", "liveClock",
                "riskScoreValue", "gaugeNeedle", "driftValue", "driftSub",
                "volValue", "volSub", "interceptValue", "interceptSub",
                "forecastBadge", "allocationBadge", "allocationFootnote",
                "velocityBadge", "ridgeBadge", "ridgeTableBody", "ridgeRSquared",
                "volMembershipLow", "volMembershipMed", "volMembershipHigh",
                "demMembershipLow", "demMembershipMed", "demMembershipHigh",
                "statRecords", "statPaths", "statPortfolioVol",
                "statExpectedReturn", "statPollInterval", "statRequestCount",
                "systemBadge",
            ];
            ids.forEach((id) => { this.els[id] = document.getElementById(id); });

            this.els.legendLow = document.querySelector(".legend-chip--low");
            this.els.legendMed = document.querySelector(".legend-chip--med");
            this.els.legendHigh = document.querySelector(".legend-chip--high");

            if (this.els.statPollInterval) {
                this.els.statPollInterval.textContent =
                    (POLL_INTERVAL_MS / 1000).toFixed(1) + "s";
            }
        }

        // -------------------------------------------------------------- //
        // LIVE CLOCK
        // -------------------------------------------------------------- //
        _startClock() {
            const tick = () => {
                if (!this.els.liveClock) return;
                const now = new Date();
                const hh = String(now.getHours()).padStart(2, "0");
                const mm = String(now.getMinutes()).padStart(2, "0");
                const ss = String(now.getSeconds()).padStart(2, "0");
                this.els.liveClock.textContent = `${hh}:${mm}:${ss}`;
            };
            tick();
            setInterval(tick, 1000);
        }

        // -------------------------------------------------------------- //
        // POLLING LIFECYCLE
        // -------------------------------------------------------------- //
        _startPolling() {
            if (this.isPolling) return;
            this.isPolling = true;
            this.pollTimerId = setInterval(() => this._pollOnce(), POLL_INTERVAL_MS);
        }

        _stopPolling() {
            if (this.pollTimerId !== null) {
                clearInterval(this.pollTimerId);
                this.pollTimerId = null;
            }
            this.isPolling = false;
        }

        /**
         * Performs a single fetch-and-render cycle against the compute
         * endpoint. Isolated as its own async method (rather than inlined
         * into the interval callback) so both the immediate boot fetch and
         * every subsequent recurring poll share identical error handling.
         */
        async _pollOnce() {
            try {
                const payload = await this._fetchQuantMatrix();
                this.requestCount += 1;
                this._setConnectionStatus(true);
                this._render(payload);
            } catch (err) {
                console.error("QuantDashboard: poll cycle failed:", err);
                this._setConnectionStatus(false, err && err.message);
            }
        }

        /**
         * Issues the network request with an explicit timeout guard (via
         * AbortController) so a hung backend computation cannot leave the
         * dashboard silently stuck "CONNECTING..." forever.
         *
         * @returns {Promise<Object>} the parsed JSON analytical payload.
         * @throws {Error} on non-2xx HTTP status, network failure, JSON
         *   parse failure, or timeout.
         */
        async _fetchQuantMatrix() {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

            try {
                const response = await fetch(COMPUTE_ENDPOINT, {
                    method: "GET",
                    headers: { "Accept": "application/json" },
                    signal: controller.signal,
                });

                if (!response.ok) {
                    let detail = `HTTP ${response.status}`;
                    try {
                        const errBody = await response.json();
                        if (errBody && errBody.message) {
                            detail += `: ${errBody.message}`;
                        }
                    } catch (_parseErr) {
                        // Response body wasn't JSON -- fall back to the
                        // bare status detail already captured above.
                    }
                    throw new Error(detail);
                }

                const data = await response.json();
                if (!data || typeof data !== "object") {
                    throw new Error("Malformed payload: expected a JSON object.");
                }
                return data;
            } catch (err) {
                if (err && err.name === "AbortError") {
                    throw new Error(`Request timed out after ${FETCH_TIMEOUT_MS}ms`);
                }
                throw err;
            } finally {
                clearTimeout(timeoutId);
            }
        }

        // -------------------------------------------------------------- //
        // CONNECTION STATUS
        // -------------------------------------------------------------- //
        _setConnectionStatus(isOnline, errorMessage) {
            const dot = this.els.connectionDot;
            const label = this.els.connectionLabel;
            if (!dot || !label) return;

            dot.classList.remove("is-online", "is-error");
            if (isOnline) {
                dot.classList.add("is-online");
                label.textContent = "STREAM ONLINE";
                if (this.els.systemBadge) {
                    this.els.systemBadge.textContent = "NOMINAL";
                }
            } else {
                dot.classList.add("is-error");
                label.textContent = errorMessage
                    ? `LINK FAULT`
                    : "DISCONNECTED";
                if (this.els.systemBadge) {
                    this.els.systemBadge.textContent = "DEGRADED";
                }
            }
        }

        // -------------------------------------------------------------- //
        // TOP-LEVEL RENDER DISPATCH
        // -------------------------------------------------------------- //
        _render(payload) {
            try {
                this._renderHeaderMeta(payload.meta);
                this._renderKpis(payload.mle, payload.ridge_regression, payload.fuzzy_risk);
                this._renderGauge(payload.fuzzy_risk);
                this._renderMembershipBars(payload.fuzzy_risk);
                this._renderRidgeTable(payload.ridge_regression);
                this._renderForecastChart(payload.sde_forecast);
                this._renderAllocationChart(payload.portfolio_allocation);
                this._renderVelocityChart(payload.raw_series);
                this._renderStatList(payload);
            } catch (err) {
                console.error("QuantDashboard: render failed on a valid payload:", err);
            }
        }

        // -------------------------------------------------------------- //
        // HEADER META
        // -------------------------------------------------------------- //
        _renderHeaderMeta(meta) {
            if (!meta) return;
            this._setText(this.els.recordCountValue, this._formatInt(meta.n_records));
            this._setText(this.els.computeTimeValue, `${meta.computation_time_ms.toFixed(1)}ms`);

            const generated = meta.generated_at_utc ? new Date(meta.generated_at_utc) : new Date();
            const hh = String(generated.getUTCHours()).padStart(2, "0");
            const mm = String(generated.getUTCMinutes()).padStart(2, "0");
            const ss = String(generated.getUTCSeconds()).padStart(2, "0");
            this._setText(this.els.lastTickValue, `${hh}:${mm}:${ss} UTC`);
        }

        // -------------------------------------------------------------- //
        // KPI STRIP
        // -------------------------------------------------------------- //
        _renderKpis(mle, ridge, fuzzy) {
            if (mle) {
                this._flashText(this.els.driftValue, this._formatSigned(mle.mu, 5));
                this._setText(
                    this.els.driftSub,
                    `&sigma;&sup2; var(log r)=${mle.variance_log_return.toExponential(2)}`,
                    true
                );
                this._flashText(this.els.volValue, `&plusmn;${(mle.sigma * 100).toFixed(3)}%`, true);
                this._setText(
                    this.els.volSub,
                    `n=${this._formatInt(mle.n_observations)} observations`
                );
            }

            if (ridge) {
                this._flashText(this.els.interceptValue, ridge.intercept.toFixed(3));
                this._setText(
                    this.els.interceptSub,
                    `R&sup2;=${ridge.r_squared.toFixed(4)} &middot; &alpha;=${ridge.alpha}`,
                    true
                );
            }
        }

        // -------------------------------------------------------------- //
        // FUZZY RISK GAUGE (signature element)
        // -------------------------------------------------------------- //
        _renderGauge(fuzzy) {
            if (!fuzzy || typeof fuzzy.risk_score !== "number") return;
            const score = Math.max(0, Math.min(100, fuzzy.risk_score));

            this._flashText(this.els.riskScoreValue, score.toFixed(1));

            // Map [0, 100] onto a [-90deg, +90deg] rotation of the needle,
            // where 0deg (unrotated) already points straight up (score 50)
            // given the SVG line's native vertical orientation.
            const angleDeg = (score / 100) * 180 - 90;
            if (this.els.gaugeNeedle) {
                this.els.gaugeNeedle.setAttribute(
                    "transform", `rotate(${angleDeg.toFixed(2)} 100 110)`
                );
            }

            [this.els.legendLow, this.els.legendMed, this.els.legendHigh].forEach((chip) => {
                if (chip) chip.classList.remove("is-active");
            });
            if (score < 33.33 && this.els.legendLow) {
                this.els.legendLow.classList.add("is-active");
            } else if (score < 66.66 && this.els.legendMed) {
                this.els.legendMed.classList.add("is-active");
            } else if (this.els.legendHigh) {
                this.els.legendHigh.classList.add("is-active");
            }
        }

        // -------------------------------------------------------------- //
        // FUZZY MEMBERSHIP BARS
        // -------------------------------------------------------------- //
        _renderMembershipBars(fuzzy) {
            if (!fuzzy) return;
            const vol = fuzzy.volatility_memberships || {};
            const dem = fuzzy.demand_memberships || {};

            this._setWidth(this.els.volMembershipLow, vol.low);
            this._setWidth(this.els.volMembershipMed, vol.medium);
            this._setWidth(this.els.volMembershipHigh, vol.high);
            this._setWidth(this.els.demMembershipLow, dem.low);
            this._setWidth(this.els.demMembershipMed, dem.medium);
            this._setWidth(this.els.demMembershipHigh, dem.high);
        }

        _setWidth(el, fraction) {
            if (!el || typeof fraction !== "number") return;
            const pct = Math.max(0, Math.min(1, fraction)) * 100;
            el.style.width = `${pct.toFixed(1)}%`;
        }

        // -------------------------------------------------------------- //
        // RIDGE REGRESSION TABLE
        // -------------------------------------------------------------- //
        _renderRidgeTable(ridge) {
            if (!ridge || !this.els.ridgeTableBody) return;

            const rows = [
                ["Intercept (&beta;&#8320;)", ridge.intercept],
                ["Matrix Distance", ridge.coefficients.matrix_distance],
                ["Passenger Velocity", ridge.coefficients.passenger_velocity],
                ["Surcharges", ridge.coefficients.surcharges],
            ];

            this.els.ridgeTableBody.innerHTML = rows.map(([label, value]) => {
                const cls = value >= 0 ? "coef-positive" : "coef-negative";
                const sign = value >= 0 ? "+" : "";
                return `<tr><td>${label}</td><td class="${cls}">${sign}${value.toFixed(5)}</td></tr>`;
            }).join("");

            this._setText(this.els.ridgeRSquared, `R&sup2; = ${ridge.r_squared.toFixed(4)}`, true);
            this._setText(this.els.ridgeBadge, `&alpha; = ${ridge.alpha}`, true);
        }

        // -------------------------------------------------------------- //
        // CHART INITIALIZATION (shells only -- populated on first render)
        // -------------------------------------------------------------- //
        _initCharts() {
            if (typeof Chart === "undefined") {
                console.error("QuantDashboard: Chart.js failed to load; charts disabled.");
                return;
            }

            Chart.defaults.font.family = COLORS.fontMono || "'JetBrains Mono', monospace";
            Chart.defaults.color = COLORS.textSecondary;

            const forecastCtx = document.getElementById("sdeForecastChart");
            if (forecastCtx) {
                this.charts.forecast = new Chart(forecastCtx, {
                    type: "line",
                    data: { labels: [], datasets: [] },
                    options: this._forecastChartOptions(),
                });
            }

            const allocationCtx = document.getElementById("allocationPieChart");
            if (allocationCtx) {
                this.charts.allocation = new Chart(allocationCtx, {
                    type: "doughnut",
                    data: { labels: [], datasets: [] },
                    options: this._allocationChartOptions(),
                });
            }

            const velocityCtx = document.getElementById("rawVelocityTickerChart");
            if (velocityCtx) {
                this.charts.velocity = new Chart(velocityCtx, {
                    type: "line",
                    data: { labels: [], datasets: [] },
                    options: this._velocityChartOptions(),
                });
            }
        }

        _sharedGridOptions() {
            return {
                grid: { color: COLORS.gridLine, drawTicks: false },
                ticks: { color: COLORS.textTertiary, font: { size: 10 } },
                border: { color: COLORS.gridLine },
            };
        }

        _forecastChartOptions() {
            return {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 420 },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: COLORS.textSecondary,
                            boxWidth: 10,
                            font: { size: 10.5 },
                            filter: (item) => !item.text.startsWith("Path "),
                        },
                    },
                    tooltip: {
                        backgroundColor: "#0f1730",
                        borderColor: "#24314f",
                        borderWidth: 1,
                        titleColor: COLORS.textPrimary,
                        bodyColor: COLORS.textSecondary,
                    },
                },
                scales: {
                    x: { ...this._sharedGridOptions(), title: { display: true, text: "Forward Step", color: COLORS.textTertiary, font: { size: 10 } } },
                    y: { ...this._sharedGridOptions(), title: { display: true, text: "Simulated Price (USD)", color: COLORS.textTertiary, font: { size: 10 } } },
                },
            };
        }

        _allocationChartOptions() {
            return {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 420 },
                cutout: "62%",
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { color: COLORS.textSecondary, boxWidth: 10, font: { size: 10.5 }, padding: 12 },
                    },
                    tooltip: {
                        backgroundColor: "#0f1730",
                        borderColor: "#24314f",
                        borderWidth: 1,
                        titleColor: COLORS.textPrimary,
                        bodyColor: COLORS.textSecondary,
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${(ctx.parsed * 100).toFixed(2)}%`,
                        },
                    },
                },
            };
        }

        _velocityChartOptions() {
            return {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#0f1730",
                        borderColor: "#24314f",
                        borderWidth: 1,
                        titleColor: COLORS.textPrimary,
                        bodyColor: COLORS.textSecondary,
                    },
                },
                scales: {
                    x: { ...this._sharedGridOptions(), ticks: { ...this._sharedGridOptions().ticks, maxTicksLimit: 6 } },
                    y: { ...this._sharedGridOptions(), title: { display: true, text: "mph", color: COLORS.textTertiary, font: { size: 10 } } },
                },
            };
        }

        // -------------------------------------------------------------- //
        // FORECAST CHART (SDE multi-path fan)
        // -------------------------------------------------------------- //
        _renderForecastChart(sde) {
            const chart = this.charts.forecast;
            if (!chart || !sde || !Array.isArray(sde.mean_path)) return;

            const stepLabels = sde.mean_path.map((_, i) => i);
            const datasets = [];

            (sde.sample_paths || []).forEach((path, idx) => {
                datasets.push({
                    label: `Path ${idx + 1}`,
                    data: path,
                    borderColor: "rgba(139, 124, 255, 0.22)",
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.15,
                    fill: false,
                });
            });

            datasets.push({
                label: "P95 Envelope",
                data: sde.p95_path,
                borderColor: "rgba(255, 180, 67, 0.55)",
                backgroundColor: COLORS.amberDim,
                borderWidth: 1,
                borderDash: [4, 3],
                pointRadius: 0,
                tension: 0.15,
                fill: "+1",
            });

            datasets.push({
                label: "P05 Envelope",
                data: sde.p05_path,
                borderColor: "rgba(255, 180, 67, 0.55)",
                borderWidth: 1,
                borderDash: [4, 3],
                pointRadius: 0,
                tension: 0.15,
                fill: false,
            });

            datasets.push({
                label: "Mean Forecast",
                data: sde.mean_path,
                borderColor: COLORS.cyan,
                backgroundColor: COLORS.cyanDim,
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.15,
                fill: false,
            });

            chart.data.labels = stepLabels;
            chart.data.datasets = datasets;
            chart.update("none");

            if (this.els.forecastBadge) {
                const pathCount = (sde.sample_paths || []).length;
                this.els.forecastBadge.textContent = `${pathCount} SAMPLE PATHS`;
            }
        }

        // -------------------------------------------------------------- //
        // ALLOCATION PIE CHART
        // -------------------------------------------------------------- //
        _renderAllocationChart(alloc) {
            const chart = this.charts.allocation;
            if (!chart || !alloc || !Array.isArray(alloc.weights)) return;

            const labels = (alloc.asset_labels || alloc.weights.map((_, i) => `Asset ${i + 1}`))
                .map((label) => label.replace(/_/g, " ").toUpperCase());

            chart.data.labels = labels;
            chart.data.datasets = [{
                data: alloc.weights,
                backgroundColor: this.assetPalette,
                borderColor: "#0b0f19",
                borderWidth: 2,
                hoverOffset: 6,
            }];
            chart.update("none");

            if (this.els.allocationFootnote) {
                this.els.allocationFootnote.textContent =
                    `&sigma;=${(alloc.portfolio_volatility * 100).toFixed(3)}% &middot; `
                        .replace("&sigma;", "vol");
                this.els.allocationFootnote.innerHTML =
                    `vol=${(alloc.portfolio_volatility * 100).toFixed(3)}% &middot; ` +
                    `E[r]=${(alloc.expected_portfolio_return * 100).toFixed(3)}%`;
            }
        }

        // -------------------------------------------------------------- //
        // RAW VELOCITY TICKER CHART
        // -------------------------------------------------------------- //
        _renderVelocityChart(rawSeries) {
            const chart = this.charts.velocity;
            if (!chart || !rawSeries || !Array.isArray(rawSeries.velocity)) return;

            const labels = (rawSeries.timestamps_ms || []).map((ms) => {
                const d = new Date(ms);
                const mm = String(d.getUTCMinutes()).padStart(2, "0");
                const ss = String(d.getUTCSeconds()).padStart(2, "0");
                return `${mm}:${ss}`;
            });

            chart.data.labels = labels;
            chart.data.datasets = [{
                label: "Passenger Velocity (mph)",
                data: rawSeries.velocity,
                borderColor: COLORS.cyan,
                backgroundColor: COLORS.cyanDim,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.25,
                fill: true,
            }];
            chart.update("none");
        }

        // -------------------------------------------------------------- //
        // SYSTEM STAT LIST
        // -------------------------------------------------------------- //
        _renderStatList(payload) {
            if (payload.meta) {
                this._setText(this.els.statRecords, this._formatInt(payload.meta.n_records));
            }
            if (payload.sde_forecast && payload.sde_forecast.sample_paths) {
                this._setText(this.els.statPaths, String(payload.sde_forecast.sample_paths.length));
            }
            if (payload.portfolio_allocation) {
                this._setText(
                    this.els.statPortfolioVol,
                    `${(payload.portfolio_allocation.portfolio_volatility * 100).toFixed(3)}%`
                );
                this._setText(
                    this.els.statExpectedReturn,
                    `${(payload.portfolio_allocation.expected_portfolio_return * 100).toFixed(3)}%`
                );
            }
            this._setText(this.els.statRequestCount, String(this.requestCount));
        }

        // -------------------------------------------------------------- //
        // FORMATTING HELPERS
        // -------------------------------------------------------------- //
        _setText(el, text, isHtml) {
            if (!el) return;
            if (isHtml) {
                el.innerHTML = text;
            } else {
                el.textContent = text;
            }
        }

        _flashText(el, text, isHtml) {
            if (!el) return;
            this._setText(el, text, isHtml);
            el.classList.remove("value-flash");
            // Force reflow so the animation re-triggers on every update,
            // not just the first time the class is added.
            void el.offsetWidth;
            el.classList.add("value-flash");
        }

        _formatInt(n) {
            if (typeof n !== "number") return "\u2014";
            return n.toLocaleString("en-US");
        }

        _formatSigned(n, decimals) {
            if (typeof n !== "number") return "\u2014";
            const sign = n >= 0 ? "+" : "";
            return `${sign}${n.toFixed(decimals)}`;
        }
    }

    // -------------------------------------------------------------------- //
    // BOOTSTRAP
    // -------------------------------------------------------------------- //
    document.addEventListener("DOMContentLoaded", () => {
        const dashboard = new QuantDashboard();
        dashboard.init();
    });
})();
