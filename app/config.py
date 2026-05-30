"""
app/config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Configuration Management

In step1_notebook.py, paths and thresholds were hardcoded:
  pd.read_csv("data/fraud_data.csv")
  n_estimators=100

Here, all configuration lives in .env and is loaded once.
Change environments (dev → staging → prod) by swapping .env,
not by editing code.

This maps to the "Configuration Management" box in the
Sculley hidden technical debt diagram.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "fraud-detection-api"
    model_name: str = "fraud_detector"
    mlflow_tracking_uri: str = "./mlruns"
    model_stage: str = "Production"
    max_amount: float = 50000.0
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Single shared instance — import this everywhere
settings = Settings()
