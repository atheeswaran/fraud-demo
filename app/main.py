"""
app/main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — FastAPI Serving Infrastructure

The notebook model is now reachable over HTTP.
Any application, team, or service can call it.

What this file adds (each is a box in the Sculley diagram):
  ✓ Serving infrastructure   — FastAPI app + endpoints
  ✓ Model loading            — loaded once at startup, shared
  ✓ Monitoring & logging     — every request is timed and logged
  ✓ Error handling           — validation errors → 422, runtime → 500
  ✓ Health check             — /health for uptime monitoring

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlflow.sklearn

from app.config import settings
from app.schemas import (
    TransactionInput, PredictionResult,
    BatchInput, BatchPredictionResult,
)
from app.pipeline import predict
from app.logger import get_logger, Timer

logger = get_logger(__name__)

# ── Model state — loaded once, reused for every request ───────
class ModelStore:
    model = None
    model_version: str = "unknown"
    run_id: str = "unknown"

store = ModelStore()


# ── Lifespan: load model at startup ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the MLflow model once when the server starts."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    try:
        store.model = mlflow.sklearn.load_model(model_uri)
        # Fetch metadata from the registry
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            settings.model_name, stages=[settings.model_stage]
        )
        if versions:
            v = versions[0]
            store.model_version = v.version
            store.run_id = v.run_id
        logger.info(
            "Model loaded",
            extra={
                "model_name":    settings.model_name,
                "model_version": store.model_version,
                "run_id":        store.run_id,
            },
        )
    except Exception as exc:
        logger.warning(
            "MLflow model not found — loading from local file",
            extra={"error": str(exc)},
        )
        # Graceful fallback for the demo if MLflow registry isn't populated
        import joblib, pathlib
        fallback = pathlib.Path("model.pkl")
        if fallback.exists():
            store.model = joblib.load(fallback)
            store.model_version = "local-fallback"
            logger.info("Loaded fallback model.pkl")
        else:
            logger.error("No model found. Run training/step2_mlflow_train.py first.")
    yield
    logger.info("Server shutting down")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "SE4ML Webinar 1 Demo — production ML service.\n\n"
        "Demonstrates: serving infrastructure · schema validation · "
        "structured logging · pipe-and-filter pattern · MLflow versioning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """
    Health check endpoint.
    Returns model version — essential for monitoring and canary deployments.

    TAKE-HOME CHALLENGE: extend this to also return last_training_timestamp.
    Hint: fetch it from mlflow.tracking.MlflowClient().get_run(store.run_id)
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":        "healthy",
        "model_name":    settings.model_name,
        "model_version": store.model_version,
        "run_id":        store.run_id,
    }


@app.post("/predict", response_model=PredictionResult, tags=["inference"])
def predict_single(txn: TransactionInput):
    """
    Predict fraud risk for a single transaction.

    Try a suspicious transaction:
      { "amount": 8500, "hour_of_day": 2, "is_foreign": 1,
        "num_transactions_last_hour": 9 }

    Try a validation failure:
      { "amount": -500, "hour_of_day": 99, "is_foreign": 2,
        "num_transactions_last_hour": 1 }
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with Timer() as t:
        try:
            result = predict(txn, store.model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Prediction error", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(
        "Prediction served",
        extra={
            "amount":       txn.amount,
            "hour_of_day":  txn.hour_of_day,
            "is_foreign":   txn.is_foreign,
            "is_fraud":     result.is_fraud,
            "risk_level":   result.risk_level,
            "latency_ms":   t.elapsed_ms,
            "model_version": store.model_version,
        },
    )
    return result


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["inference"])
def predict_batch(payload: BatchInput):
    """
    Predict fraud risk for up to 100 transactions in one call.
    Demonstrates batch serving — relevant for nightly fraud sweeps.
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with Timer() as t:
        results = []
        for txn in payload.transactions:
            try:
                results.append(predict(txn, store.model))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

    flagged = sum(1 for r in results if r.is_fraud)

    logger.info(
        "Batch prediction served",
        extra={
            "batch_size": len(payload.transactions),
            "flagged":    flagged,
            "latency_ms": t.elapsed_ms,
        },
    )

    return BatchPredictionResult(
        predictions=results,
        total=len(results),
        flagged=flagged,
    )
