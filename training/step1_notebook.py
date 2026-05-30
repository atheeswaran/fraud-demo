"""
training/step1_notebook.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — The Notebook

This is where most ML work starts. It works. It gets results.
But look at what's missing:

  ✗ No schema validation on the input data
  ✗ No error handling — crashes silently on bad data
  ✗ No logging — impossible to debug in production
  ✗ Hardcoded file paths — breaks on any other machine
  ✗ No model versioning — which model is in production?
  ✗ No experiment tracking — can't compare runs
  ✗ No serving — nobody else can call this model
  ✗ No tests — no confidence in the output

This is the "ML code is a tiny box" reality from the Sculley diagram.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data — hardcoded path
df = pd.read_csv("data/fraud_data.csv")

# Features and label
X = df[["amount", "hour_of_day", "is_foreign", "num_transactions_last_hour"]]
y = df["is_fraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model — hyperparameters hardcoded
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save — no versioning, just overwrites
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")

# ── PRESENTER NOTE ────────────────────────────────────────────
# Ask students: "What happens when this model degrades in 3 months?"
# "How do you know which version of model.pkl is in production?"
# "If this crashes at 2am, how do you debug it?"
# These questions motivate every file in the /app directory.
# ─────────────────────────────────────────────────────────────
