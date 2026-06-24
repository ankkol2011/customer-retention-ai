"""CSV parsing and upload validation helpers."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from frontend.src.churn_ui.schemas import UploadValidationResult


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read a Streamlit upload into a dataframe."""
    if uploaded_file is None:
        raise ValueError("No CSV file was uploaded.")

    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8-sig")
    return pd.read_csv(StringIO(text))


def validate_uploaded_csv(data: pd.DataFrame, *, require_target: bool = False) -> UploadValidationResult:
    """Apply lightweight upload checks before sending data to the scorer."""
    required_columns = [
        "customer_id",
        "signup_date",
        "age",
        "gender",
        "city",
        "education_level",
        "employment_status",
        "monthly_income",
        "monthly_bill",
        "internet_usage_gb",
        "call_minutes",
        "contract_type",
        "support_tickets",
        "customer_feedback",
    ]
    if require_target:
        required_columns.append("churn")

    missing_columns = [column for column in required_columns if column not in data.columns]
    messages: list[str] = []
    if missing_columns:
        messages.append(f"Missing required columns: {', '.join(missing_columns)}")

    if data.empty:
        messages.append("CSV file is empty.")

    duplicate_customer_ids = int(data["customer_id"].duplicated().sum()) if "customer_id" in data.columns else 0
    if duplicate_customer_ids:
        messages.append(f"Found {duplicate_customer_ids} duplicate customer_id values.")

    if "churn" in data.columns and require_target:
        target_values = pd.to_numeric(data["churn"], errors="coerce")
        invalid_target_rows = int(target_values.isna().sum())
        if invalid_target_rows:
            messages.append("Some churn values are not numeric 0/1 values.")

    return UploadValidationResult(
        passed=not messages,
        messages=messages,
        row_count=int(len(data)),
        column_count=int(len(data.columns)),
        columns=list(data.columns),
    )
