"""
app.py
==============================================================================
THE ENTERPRISE CORE CONTROLLER GATEWAY
==============================================================================

Flask REST API layer (Controller) for the High-Frequency Macro Algorithmic
Arbitrage Engine. This module deliberately contains zero mathematical logic
-- every quantitative computation is delegated to `quant_model.AdvancedQuantEngine`
(the Model layer). app.py's job is strictly:

    - process lifecycle: instantiate and pre-warm the engine once at boot,
      before the first request is ever served;
    - concurrency: offload each heavy analytical request onto a bounded
      worker thread pool so the Flask request-handling thread is never
      blocked on numpy/scipy computation directly;
    - transport: translate engine output into clean, minified JSON and
      engine failures into well-formed HTTP error responses;
    - observability: wrap every route in structured request/response
      logging (method, path, remote address, status, duration).

Run directly for local development:
    python app.py

For production, run behind a real WSGI server, e.g.:
    gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
==============================================================================
"""

import functools
import logging
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from flask import Flask, jsonify, render_template, request

from quant_model import AdvancedQuantEngine, QuantEngineError

# --------------------------------------------------------------------------- #
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------- #
# Configured once, at module import time, so both the Werkzeug dev server
# and any production WSGI runner (gunicorn, uwsgi) that imports this module
# inherit identical structured formatting.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("quant_engine.controller")

# --------------------------------------------------------------------------- #
# APPLICATION + CONCURRENCY POOL SETUP
# --------------------------------------------------------------------------- #
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

# Bounded worker pool: analytical requests are CPU-bound (numpy/scipy linear
# algebra), so a modest fixed pool prevents unbounded thread creation under
# concurrent polling load from the frontend's high-frequency refresh loop,
# while still keeping the Flask request thread free to serve other routes
# (health checks, static assets) while a computation is in flight.
COMPUTE_POOL_MAX_WORKERS = 4
COMPUTE_TIMEOUT_SECONDS = 30
compute_executor = ThreadPoolExecutor(
    max_workers=COMPUTE_POOL_MAX_WORKERS,
    thread_name_prefix="quant-compute-worker",
)

# --------------------------------------------------------------------------- #
# ENGINE INSTANTIATION + PRE-WARM
# --------------------------------------------------------------------------- #
# The engine is instantiated at import time and pre-warmed immediately --
# i.e. its multi-gigabyte-style synthetic dataset pool is generated once,
# eagerly, before the application ever accepts a request. This guarantees
# the very first inbound HTTP request never pays the (relatively) expensive
# dataset-generation cost synchronously inside the request/response cycle.
# --------------------------------------------------------------------------- #
quant_engine = AdvancedQuantEngine(
    n_records=125_000,
    n_sde_paths=200,
    n_sde_steps=252,
    random_seed=42,
)

try:
    _boot_start = time.perf_counter()
    quant_engine.warm_up()
    _boot_elapsed = time.perf_counter() - _boot_start
    logger.info(
        "Boot sequence complete: AdvancedQuantEngine pre-warmed with %d "
        "synthetic records in %.4fs.",
        quant_engine.n_records, _boot_elapsed,
    )
except QuantEngineError:
    # A failure to warm up the engine at boot is fatal -- there is no
    # meaningful degraded mode for a quant engine with no data. Log the
    # full trace and re-raise so the process exits non-zero and any
    # process supervisor (systemd, gunicorn, docker healthcheck) sees a
    # clean, immediate failure rather than a "successfully" running
    # server that 500s on every request.
    logger.critical(
        "FATAL: engine pre-warm failed at boot. Aborting startup.\n%s",
        traceback.format_exc(),
    )
    raise


# --------------------------------------------------------------------------- #
# COMPREHENSIVE REQUEST/RESPONSE LOGGING DECORATOR
# --------------------------------------------------------------------------- #
def logged_route(route_name):
    """
    Decorator factory producing a structured logging wrapper around a Flask
    view function. Every wrapped route logs:

        - a unique per-request correlation id (for tracing a single
          request across multiple log lines / downstream services),
        - inbound method, path, and remote address,
        - outcome (status code) and wall-clock duration in milliseconds,
        - full exception detail on unhandled failures, before re-raising
          so Flask's own error handling still produces the HTTP response.

    Args:
        route_name (str): a short, human-readable label for the wrapped
            route, included in every log line to make route-level
            filtering trivial in log aggregation tooling.

    Returns:
        Callable: the actual decorator to apply to a Flask view function.
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            start_ts = time.perf_counter()
            logger.info(
                "[%s] --> %s %s route=%s remote_addr=%s",
                request_id, request.method, request.path, route_name,
                request.remote_addr,
            )
            try:
                response = view_func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

                # Normalize Flask's various legal return shapes (Response
                # object, (body, status) tuple, plain body) to extract a
                # status code for logging without altering what's returned.
                if isinstance(response, tuple) and len(response) >= 2:
                    status_code = response[1]
                else:
                    status_code = getattr(response, "status_code", 200)

                logger.info(
                    "[%s] <-- %s %s status=%s duration_ms=%.3f",
                    request_id, request.method, request.path,
                    status_code, elapsed_ms,
                )
                return response
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                logger.error(
                    "[%s] xx  %s %s FAILED after %.3fms: %s\n%s",
                    request_id, request.method, request.path, elapsed_ms,
                    exc, traceback.format_exc(),
                )
                raise
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# ROUTES
# --------------------------------------------------------------------------- #
@app.route("/", methods=["GET"])
@logged_route("dashboard_shell")
def dashboard_index():
    """
    Serves the primary dashboard application shell view -- the static HTML
    skeleton that the client-side ES6 dashboard engine (`dashboard.js`)
    subsequently populates via polling calls to `/api/compute-quant-matrix`.

    Returns:
        flask.Response: rendered `templates/index.html`.
    """
    try:
        return render_template("index.html")
    except Exception as exc:
        logger.error("dashboard_index failed to render template: %s", exc)
        return jsonify({
            "error": "TEMPLATE_RENDER_FAILURE",
            "message": "The dashboard shell could not be rendered.",
        }), 500


@app.route("/api/compute-quant-matrix", methods=["GET"])
@logged_route("compute_quant_matrix")
def compute_quant_matrix():
    """
    Asynchronously computes the full quantitative analytical matrix:

        - MLE-parsed stochastic calculus drift/volatility from the live
          time-series slice,
        - the multi-path Euler-Maruyama SDE forecast array,
        - custom closed-form ridge regression coefficients,
        - the constrained multivariable minimum-variance portfolio
          allocation,
        - the Mamdani fuzzy-logic composite Risk Asset Score,

    and returns them as a single, minified JSON payload.

    The actual computation is submitted to `compute_executor` (a bounded
    ThreadPoolExecutor) rather than executed inline, so that CPU-bound
    numpy/scipy work never blocks the Flask request-handling thread beyond
    the `.result()` wait, and so a slow computation cannot starve other
    concurrently-arriving requests (health checks, dashboard shell loads).

    Query Parameters:
        ridge_alpha (float, optional): overrides the default L2
            regularization strength for the ridge regression subsystem.

    Returns:
        flask.Response: JSON payload (200) on success; a structured JSON
        error payload (400/500/504) on failure.
    """
    try:
        ridge_alpha_param = request.args.get("ridge_alpha", default=1.0, type=float)
        if ridge_alpha_param is None or ridge_alpha_param < 0:
            return jsonify({
                "error": "INVALID_PARAMETER",
                "message": "ridge_alpha must be a non-negative float.",
            }), 400

        future = compute_executor.submit(
            quant_engine.run_full_analysis, ridge_alpha_param
        )

        try:
            payload = future.result(timeout=COMPUTE_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            logger.error(
                "compute_quant_matrix: analytical pass exceeded %ds timeout.",
                COMPUTE_TIMEOUT_SECONDS,
            )
            return jsonify({
                "error": "COMPUTE_TIMEOUT",
                "message": (
                    f"Analytical computation exceeded the "
                    f"{COMPUTE_TIMEOUT_SECONDS}s execution budget."
                ),
            }), 504

        response = jsonify(payload)
        response.status_code = 200
        return response

    except QuantEngineError as exc:
        logger.error("compute_quant_matrix: engine-level failure: %s", exc)
        return jsonify({
            "error": "QUANT_ENGINE_FAILURE",
            "message": str(exc),
        }), 500
    except Exception as exc:
        logger.error(
            "compute_quant_matrix: unexpected failure: %s\n%s",
            exc, traceback.format_exc(),
        )
        return jsonify({
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred while computing the "
                       "quantitative matrix.",
        }), 500


@app.route("/api/health", methods=["GET"])
@logged_route("health_check")
def health_check():
    """
    Lightweight liveness/readiness probe, independent of the heavy compute
    pool, so orchestration tooling (load balancers, container health
    checks) can distinguish "process is alive and engine is warm" from a
    genuinely overloaded or crashed analytical subsystem.

    Returns:
        flask.Response: JSON status payload (200).
    """
    return jsonify({
        "status": "OK",
        "engine_warm": bool(quant_engine._is_warmed),
        "n_records": quant_engine.n_records,
        "compute_pool_max_workers": COMPUTE_POOL_MAX_WORKERS,
    }), 200


# --------------------------------------------------------------------------- #
# ERROR HANDLERS
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def handle_not_found(err):
    """Uniform JSON 404 payload for any unmatched route."""
    return jsonify({
        "error": "NOT_FOUND",
        "message": f"The requested resource '{request.path}' does not exist.",
    }), 404


@app.errorhandler(405)
def handle_method_not_allowed(err):
    """Uniform JSON 405 payload for a valid route hit with the wrong verb."""
    return jsonify({
        "error": "METHOD_NOT_ALLOWED",
        "message": f"Method '{request.method}' is not allowed on "
                   f"'{request.path}'.",
    }), 405


@app.errorhandler(500)
def handle_internal_error(err):
    """
    Uniform JSON 500 payload as a last-resort catch-all. Route-level
    try/except blocks are expected to catch and translate almost every
    failure themselves; this handler exists purely as a safety net against
    anything that slips through (e.g. a failure inside Flask's own
    request-teardown machinery).
    """
    logger.error("Unhandled 500 error: %s", err)
    return jsonify({
        "error": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected server error occurred.",
    }), 500


# --------------------------------------------------------------------------- #
# ENTRYPOINT
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info(
        "Starting High-Frequency Macro Algorithmic Arbitrage Engine "
        "development server..."
    )
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
