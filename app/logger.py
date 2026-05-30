"""
app/logger.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Structured Logging

In the notebook: print("Model saved to model.pkl")
In production: structured JSON logs that monitoring tools can parse.

Structured logs let you answer questions like:
  "Show me all HIGH-risk predictions made after midnight"
  "What was the average latency yesterday?"
  "Which model version handled the spike on Friday?"

This maps to the "Monitoring & Logging" box in the Sculley diagram.

PRESENTER: Run the server and watch the terminal.
Each request produces a JSON-structured log line, not a plain string.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
import json
import time
from app.config import settings


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregation tools."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":     record.levelname,
            "service":   settings.app_name,
            "message":   record.getMessage(),
        }
        # Merge any extra fields passed via logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "message", "asctime",
            ):
                log_obj[key] = value
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class Timer:
    """Simple context manager for measuring elapsed time in ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
