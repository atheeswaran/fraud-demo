"""
training/step2_mlflow_train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Adding MLflow Experiment Tracking

Diff from step1_notebook.py — only 4 things changed:
  + mlflow.set_experiment()       → named experiment bucket
  + mlflow.start_run()            → wraps training in a tracked run
  + mlflow.log_params()           → records hyperparameters
  + mlflow.log_metrics()          → records evaluation results
  + mlflow.sklearn.log_model()    → versions the model artifact

The training logic is IDENTICAL. This is the key insight:
adding production engineering doesn't mean rewriting your ML.

Run this twice with different n_estimators to show the UI comparison:
  python training/step2_mlflow_train.py              (n_estimators=100)
  python training/step2_mlflow_train.py --trees 200  (n_estimators=200)

Then open:  mlflow ui   →  http://127.0.0.1:5000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.sklearn

# ── CLI args so presenter can run two experiments live ────────
parser = argparse.ArgumentParser()
parser.add_argument("--trees",    type=int,   default=100)
parser.add_argument("--depth",    type=int,   default=5)
parser.add_argument("--data",     type=str,   default="data/fraud_data.csv")
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────
df = pd.read_csv(args.data)
X = df[["amount", "hour_of_day", "is_foreign", "num_transactions_last_hour"]]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MLflow experiment ─────────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("fraud-detection")          # NEW: named bucket

with mlflow.start_run():                          # NEW: tracked run
    # Log hyperparameters
    mlflow.log_params({                           # NEW: record what we tried
        "n_estimators": args.trees,
        "max_depth":    args.depth,
        "random_state": 42,
    })

    # Train (unchanged from step 1)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    mlflow.log_metrics(metrics)                   # NEW: record results

    # Log & version the model artifact
    mlflow.sklearn.log_model(                     # NEW: versioned artifact
        sk_model=model,
        artifact_path="fraud_model",
        registered_model_name="fraud_detector",
        input_example=X_test.iloc[:3],
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ Run logged: {run_id}")
    print(f"  n_estimators={args.trees}, max_depth={args.depth}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nOpen the MLflow UI:  mlflow ui  →  http://127.0.0.1:5000")

# ── PRESENTER NOTE ────────────────────────────────────────────
# In the MLflow UI, show students:
#   1. Both runs appear in the experiment table
#   2. Click "Compare" — metrics side by side
#   3. Click a run → Artifacts tab → the model is versioned, not overwritten
#   4. Model Registry tab → fraud_detector versions
#
# Ask: "In step 1, which model was in production.pkl? You don't know."
# "In step 2, you always know — run ID, params, metrics, artifact path."
# ─────────────────────────────────────────────────────────────
