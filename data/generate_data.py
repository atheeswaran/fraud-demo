"""
data/generate_data.py
Generates a synthetic credit card fraud dataset for the demo.
Run once before the webinar: python data/generate_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 2000

# Features: amount, hour_of_day, is_foreign, num_transactions_last_hour
amount = np.concatenate([
    np.random.exponential(scale=100, size=int(N * 0.95)),  # legit: small amounts
    np.random.exponential(scale=2000, size=int(N * 0.05)), # fraud: large amounts
])[:N]

hour_of_day = np.concatenate([
    np.random.choice(range(8, 22), size=int(N * 0.95)),    # legit: daytime
    np.random.choice(range(0, 6),  size=int(N * 0.05)),    # fraud: late night
])[:N]

is_foreign = np.concatenate([
    np.random.binomial(1, 0.1, size=int(N * 0.95)),        # legit: mostly domestic
    np.random.binomial(1, 0.8, size=int(N * 0.05)),        # fraud: mostly foreign
])[:N]

num_txn_last_hour = np.concatenate([
    np.random.poisson(lam=1, size=int(N * 0.95)),          # legit: low frequency
    np.random.poisson(lam=8, size=int(N * 0.05)),          # fraud: high frequency
])[:N]

# Label: fraud if amount > 1000 AND (is_foreign OR hour < 6) OR many txns
is_fraud = (
    ((amount > 1000) & ((is_foreign == 1) | (hour_of_day < 6))) |
    (num_txn_last_hour > 6)
).astype(int)

df = pd.DataFrame({
    "amount": np.round(amount, 2),
    "hour_of_day": hour_of_day,
    "is_foreign": is_foreign,
    "num_transactions_last_hour": num_txn_last_hour,
    "is_fraud": is_fraud,
})

out_path = Path(__file__).parent / "fraud_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(f"Fraud rate: {is_fraud.mean():.1%}")
print(df.head())
