"""
app/schemas.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Input & Output Schema Validation

In the notebook, the model blindly accepted whatever pandas
loaded from the CSV. In production:
  - A client sends JSON. We must validate it.
  - Bad input should fail FAST with a clear error message,
    not silently produce a garbage prediction.

Pydantic enforces:
  ✓ Field types (amount must be a float, not a string)
  ✓ Value ranges (amount > 0, hour 0–23, is_foreign 0 or 1)
  ✓ Clear error messages returned to the caller

This maps to the "Data Validation" box in the Sculley diagram
and implements the ROBUSTNESS quality attribute from Session 4.

PRESENTER: Try sending a bad request in /docs:
  { "amount": -500, "hour_of_day": 99, "is_foreign": 2,
    "num_transactions_last_hour": 5 }
Watch Pydantic reject all three violations with field-level errors.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """A single transaction to score for fraud risk."""
    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount in INR. Must be positive.",
        examples=[1250.00],
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the transaction (0–23, 24-hour clock).",
        examples=[14],
    )
    is_foreign: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the transaction originates from a foreign country, else 0.",
        examples=[0],
    )
    num_transactions_last_hour: int = Field(
        ...,
        ge=0,
        description="Number of transactions on this card in the past hour.",
        examples=[2],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 1250.00,
                    "hour_of_day": 14,
                    "is_foreign": 0,
                    "num_transactions_last_hour": 2,
                }
            ]
        }
    }


class BatchInput(BaseModel):
    """A list of transactions to score in one call."""
    transactions: list[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Between 1 and 100 transactions per batch.",
    )


class PredictionResult(BaseModel):
    """Fraud prediction for a single transaction."""
    is_fraud: bool = Field(..., description="True if the model predicts fraud.")
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence that this transaction is fraudulent.",
    )
    risk_level: str = Field(
        ...,
        description="Human-readable risk tier: LOW, MEDIUM, or HIGH.",
    )


class BatchPredictionResult(BaseModel):
    """Results for a batch prediction call."""
    predictions: list[PredictionResult]
    total: int
    flagged: int
