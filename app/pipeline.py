"""
app/pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Pipe-and-Filter Pattern in Code

Session 4 introduced the pipe-and-filter architectural pattern:
    Data Source → Filter 1 → Filter 2 → Filter 3 → Data Sink

Here it is as Python:
    validate_input → extract_features → run_model → format_response

Each stage is a PURE FUNCTION:
  - Takes well-typed input
  - Returns well-typed output
  - Has NO side effects (no DB calls, no logging inside)
  - Is independently testable

This is the architectural pattern materialised in code.

PRESENTER: Draw this on a whiteboard while showing the file:
  [TransactionInput] → [dict/list] → [ndarray] → [PredictionResult]
  validate_input   extract_features  run_model   format_response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from app.schemas import TransactionInput, PredictionResult
from app.config import settings


# ── Stage 1: Validate ─────────────────────────────────────────
# Pydantic already handles this (see schemas.py).
# This stage adds one business rule that's too dynamic for a schema:
# flagging unusually high amounts as suspicious before the model runs.

def validate_input(txn: TransactionInput) -> TransactionInput:
    """
    Business-level validation beyond schema types.
    Raises ValueError for amounts that exceed the configured ceiling.
    """
    if txn.amount > settings.max_amount:
        raise ValueError(
            f"Amount {txn.amount} exceeds maximum allowed "
            f"({settings.max_amount}). Flag for manual review."
        )
    return txn


# ── Stage 2: Feature Extraction ───────────────────────────────
# Converts a validated domain object into the numeric array the
# model was trained on. The feature ORDER must match training.

FEATURE_NAMES = [
    "amount",
    "hour_of_day",
    "is_foreign",
    "num_transactions_last_hour",
]

def extract_features(txn: TransactionInput) -> np.ndarray:
    """
    Converts a TransactionInput into a (1, 4) numpy array.
    Feature order MUST match what was used in training.
    """
    return np.array([[
        txn.amount,
        txn.hour_of_day,
        txn.is_foreign,
        txn.num_transactions_last_hour,
    ]])


# ── Stage 3: Model Inference ──────────────────────────────────
# Pure inference — no business logic, just the model.

def run_model(
    features: np.ndarray,
    model: RandomForestClassifier,
) -> tuple[bool, float]:
    """
    Runs the model. Returns (is_fraud: bool, fraud_probability: float).
    Keeping inference separate from business logic makes this
    swappable: change the model without touching anything else.
    """
    proba = model.predict_proba(features)[0]
    fraud_class_index = list(model.classes_).index(1)
    fraud_prob = float(proba[fraud_class_index])
    is_fraud = fraud_prob >= 0.5
    return is_fraud, fraud_prob


# ── Stage 4: Format Response ──────────────────────────────────
# Converts raw model output into the API contract (PredictionResult).
# Business thresholds for risk tiers live here, not inside the model.

def format_response(is_fraud: bool, fraud_prob: float) -> PredictionResult:
    """
    Applies business rules to produce a human-readable risk tier.
    Thresholds are a BUSINESS decision, not an ML decision.
    """
    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return PredictionResult(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
    )


# ── Assembled Pipeline ────────────────────────────────────────
# The four stages wired together as a single callable.

def predict(
    txn: TransactionInput,
    model: RandomForestClassifier,
) -> PredictionResult:
    """
    Full pipe-and-filter pipeline.
    Called by the FastAPI route — the route itself stays thin.
    """
    txn      = validate_input(txn)           # Stage 1
    features = extract_features(txn)         # Stage 2
    is_fraud, prob = run_model(features, model)  # Stage 3
    return format_response(is_fraud, prob)   # Stage 4
