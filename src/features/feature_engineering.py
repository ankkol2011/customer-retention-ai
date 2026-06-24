"""Shared feature engineering logic for telecom churn models.

This module keeps the feature derivations pure and reusable so training and
batch inference use the same rules.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import partial
from typing import Any

import pandas as pd


def add_derived_features(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    reference_date: str | date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a copy of ``data`` with derived churn features added.

    The raw dataset has a signup date, support tickets, and free-text feedback.
    This function turns those fields into model-ready numeric features.
    """
    feature_config = config["features"]
    derived_config = feature_config["derived"]
    baseline_config = config["baseline"]

    signup_date_column = feature_config.get("date", ["signup_date"])[0]
    support_tickets_column = "support_tickets"
    feedback_column = feature_config.get("text", ["customer_feedback"])[0]

    engineered = data.copy()
    engineered[derived_config["tenure_column"]] = derive_customer_tenure_days(
        engineered,
        signup_date_column=signup_date_column,
        reference_date=reference_date,
    )
    engineered[derived_config["support_ticket_rule_column"]] = derive_support_ticket_flag(
        engineered,
        support_tickets_column=support_tickets_column,
        threshold=int(baseline_config["support_tickets_threshold"]),
    )
    feedback_features = derive_customer_feedback_features(
        engineered,
        feedback_column=feedback_column,
        keyword_prefix=derived_config.get("feedback_keyword_prefix", "customer_feedback_keyword"),
    )
    for column in feedback_features.columns:
        engineered[column] = feedback_features[column]
    return engineered


def derive_customer_tenure_days(
    data: pd.DataFrame,
    *,
    signup_date_column: str = "signup_date",
    reference_date: str | date | datetime | pd.Timestamp | None = None,
) -> pd.Series:
    """Derive a tenure-like day count from signup dates.

    If a reference date is provided, tenure is measured relative to that date.
    Otherwise the most recent signup date in the batch is used. That keeps the
    feature computable for CSV batches, while still making the reference point
    explicit when the caller knows the real observation date.
    """
    if signup_date_column not in data.columns:
        raise KeyError(f"Missing required column: {signup_date_column}")

    parsed_signup = pd.to_datetime(data[signup_date_column], errors="coerce")
    if parsed_signup.isna().any():
        invalid_values = sorted(set(data.loc[parsed_signup.isna(), signup_date_column].astype(str)))
        raise ValueError(
            f"Column {signup_date_column} contains invalid date values: {invalid_values}"
        )

    resolved_reference_date = _resolve_reference_date(reference_date, parsed_signup)
    tenure_days = (resolved_reference_date - parsed_signup).dt.days
    if (tenure_days < 0).any():
        raise ValueError(
            f"Reference date {resolved_reference_date.date()} is earlier than at least one "
            f"value in {signup_date_column}"
        )

    return tenure_days.astype("int64")


def derive_support_ticket_flag(
    data: pd.DataFrame,
    *,
    support_tickets_column: str = "support_tickets",
    threshold: int = 3,
) -> pd.Series:
    """Flag customers whose support tickets meet or exceed the threshold."""
    if support_tickets_column not in data.columns:
        raise KeyError(f"Missing required column: {support_tickets_column}")

    support_tickets = pd.to_numeric(data[support_tickets_column], errors="coerce")
    if support_tickets.isna().any():
        invalid_values = sorted(set(data.loc[support_tickets.isna(), support_tickets_column].astype(str)))
        raise ValueError(
            f"Column {support_tickets_column} contains non-numeric values: {invalid_values}"
        )

    return (support_tickets >= threshold).astype(int)


def derive_customer_feedback_features(
    data: pd.DataFrame,
    *,
    feedback_column: str = "customer_feedback",
    keyword_prefix: str = "customer_feedback_keyword",
) -> pd.DataFrame:
    """Create lightweight, explainable text features from customer feedback."""
    if feedback_column not in data.columns:
        raise KeyError(f"Missing required column: {feedback_column}")

    feedback = data[feedback_column].fillna("").astype("string")
    lowered = feedback.str.lower()

    return pd.DataFrame(
        {
            f"{keyword_prefix}_billing": lowered.str.contains("billing", regex=False).astype(int),
            f"{keyword_prefix}_coverage": lowered.str.contains("coverage", regex=False).astype(int),
            f"{keyword_prefix}_support": lowered.str.contains("support", regex=False).astype(int),
            "customer_feedback_length": feedback.str.len().astype(int),
        },
        index=data.index,
    )


def _resolve_reference_date(
    reference_date: str | date | datetime | pd.Timestamp | None,
    parsed_signup: pd.Series,
) -> pd.Timestamp:
    if reference_date is None:
        resolved = parsed_signup.max()
        if pd.isna(resolved):
            raise ValueError("Cannot derive tenure because signup dates are all invalid")
        return pd.Timestamp(resolved).normalize()

    resolved = pd.Timestamp(reference_date)
    if pd.isna(resolved):
        raise ValueError("reference_date could not be parsed into a valid timestamp")
    return resolved.normalize()


__all__ = [
    "add_derived_features",
    "derive_customer_feedback_features",
    "derive_customer_tenure_days",
    "derive_support_ticket_flag",
]
