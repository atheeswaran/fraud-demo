"""
tests/test_quality_attrs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — Tests That Encode Quality Attributes

Session 4 defined quality attributes as "measurable or testable
properties." That word — TESTABLE — is the key insight here.

Every test below is a quality attribute from Session 4 turned into
an assertion that FAILS if the system violates it:

  test_schema_rejects_bad_input   → ROBUSTNESS
  test_predict_returns_contract   → RELIABILITY
  test_latency_under_200ms        → PERFORMANCE
  test_batch_all_scored           → RELIABILITY
  test_pipeline_stages_isolated   → MAINTAINABILITY

PRESENTER: Run `pytest tests/ -v` and show the output.
Then deliberately break a threshold (e.g. change 200 to 1) and
show a test fail. This is what "quality attributes drive design" means.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.schemas import TransactionInput
from app.pipeline import (
    validate_input, extract_features, run_model, format_response, predict
)

# ── Shared fixtures ───────────────────────────────────────────

@pytest.fixture
def legit_txn():
    return TransactionInput(
        amount=500.0,
        hour_of_day=14,
        is_foreign=0,
        num_transactions_last_hour=1,
    )

@pytest.fixture
def fraud_txn():
    return TransactionInput(
        amount=9500.0,
        hour_of_day=3,
        is_foreign=1,
        num_transactions_last_hour=10,
    )

@pytest.fixture
def trained_model():
    """
    Train a minimal model for unit tests — no MLflow, no disk I/O.
    Tests are self-contained and don't need the server running.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/fraud_data.csv")
    X = df[["amount", "hour_of_day", "is_foreign", "num_transactions_last_hour"]]
    y = df["is_fraud"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def client(trained_model):
    """HTTP test client with the trained model injected."""
    from app.main import app, store
    store.model = trained_model
    store.model_version = "test-v1"
    return TestClient(app)


# ── Quality Attribute: ROBUSTNESS ─────────────────────────────

class TestRobustness:
    """The system rejects invalid input with clear error messages."""

    def test_schema_rejects_negative_amount(self):
        with pytest.raises(Exception):
            TransactionInput(
                amount=-100.0, hour_of_day=10,
                is_foreign=0, num_transactions_last_hour=1,
            )

    def test_schema_rejects_invalid_hour(self):
        with pytest.raises(Exception):
            TransactionInput(
                amount=500.0, hour_of_day=99,
                is_foreign=0, num_transactions_last_hour=1,
            )

    def test_schema_rejects_invalid_is_foreign(self):
        with pytest.raises(Exception):
            TransactionInput(
                amount=500.0, hour_of_day=10,
                is_foreign=5, num_transactions_last_hour=1,
            )

    def test_business_rule_rejects_excessive_amount(self, legit_txn):
        over_limit = TransactionInput(
            amount=99999.0, hour_of_day=14,
            is_foreign=0, num_transactions_last_hour=1,
        )
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_input(over_limit)

    def test_api_returns_422_on_bad_input(self, client):
        resp = client.post("/predict", json={"amount": -1, "hour_of_day": 99,
                                              "is_foreign": 0, "num_transactions_last_hour": 1})
        assert resp.status_code == 422


# ── Quality Attribute: RELIABILITY ────────────────────────────

class TestReliability:
    """The system produces consistent, well-formed outputs."""

    def test_predict_returns_valid_contract(self, legit_txn, trained_model):
        result = predict(legit_txn, trained_model)
        assert isinstance(result.is_fraud, bool)
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_risk_level_matches_probability(self, legit_txn, trained_model):
        result = predict(legit_txn, trained_model)
        if result.fraud_probability < 0.3:
            assert result.risk_level == "LOW"
        elif result.fraud_probability < 0.6:
            assert result.risk_level == "MEDIUM"
        else:
            assert result.risk_level == "HIGH"

    def test_batch_scores_all_transactions(self, client):
        batch = {
            "transactions": [
                {"amount": 500.0,  "hour_of_day": 14, "is_foreign": 0, "num_transactions_last_hour": 1},
                {"amount": 8500.0, "hour_of_day": 3,  "is_foreign": 1, "num_transactions_last_hour": 9},
                {"amount": 200.0,  "hour_of_day": 11, "is_foreign": 0, "num_transactions_last_hour": 0},
            ]
        }
        resp = client.post("/predict/batch", json=batch)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_health_check_returns_model_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "model_version" in resp.json()


# ── Quality Attribute: PERFORMANCE ────────────────────────────

class TestPerformance:
    """
    Single prediction must complete under 200ms.
    (Quality attribute: Performance — from Session 4)

    If this test fails, investigate:
      - Is the model too large? → consider model compression
      - Is feature extraction inefficient? → profile extract_features()
      - Is cold-start the issue? → ensure model is pre-loaded
    """

    LATENCY_THRESHOLD_MS = 200

    def test_single_prediction_latency(self, legit_txn, trained_model):
        # Warm-up call (model load, JIT)
        predict(legit_txn, trained_model)

        # Measured call
        start = time.perf_counter()
        predict(legit_txn, trained_model)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.LATENCY_THRESHOLD_MS, (
            f"Prediction took {elapsed_ms:.1f}ms — "
            f"exceeds {self.LATENCY_THRESHOLD_MS}ms SLA. "
            "Investigate model size or feature extraction."
        )


# ── Quality Attribute: MAINTAINABILITY ────────────────────────

class TestMaintainability:
    """
    Each pipeline stage is independently testable.
    This is the property that the pipe-and-filter pattern buys us.
    """

    def test_extract_features_returns_correct_shape(self, legit_txn):
        features = extract_features(legit_txn)
        assert features.shape == (1, 4), \
            "Feature array must be (1, 4) — 1 sample, 4 features"

    def test_extract_features_correct_order(self, legit_txn):
        features = extract_features(legit_txn)
        assert features[0][0] == legit_txn.amount
        assert features[0][1] == legit_txn.hour_of_day
        assert features[0][2] == legit_txn.is_foreign
        assert features[0][3] == legit_txn.num_transactions_last_hour

    def test_format_response_low_risk(self):
        result = format_response(is_fraud=False, fraud_prob=0.1)
        assert result.risk_level == "LOW"

    def test_format_response_high_risk(self):
        result = format_response(is_fraud=True, fraud_prob=0.85)
        assert result.risk_level == "HIGH"

    def test_run_model_returns_probability_in_range(self, legit_txn, trained_model):
        features = extract_features(legit_txn)
        _, prob = run_model(features, trained_model)
        assert 0.0 <= prob <= 1.0
