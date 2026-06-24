"""Configuration values for the churn review frontend."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "telecom_churn_classifier.joblib"
DEFAULT_SCORING_CSV = ROOT_DIR / "data" / "telecom_customer_churn_feature_engineering.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_REPORTS_DIR = DEFAULT_OUTPUT_DIR / "reports"

