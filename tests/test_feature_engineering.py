"""Tests for shared churn feature engineering."""

from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import (
    add_derived_features,
    derive_customer_feedback_features,
    derive_customer_tenure_days,
    derive_support_ticket_flag,
)


def _sample_config() -> dict:
    return {
        "data": {
            "id_column": "customer_id",
            "target_column": "churn",
        },
        "features": {
            "numeric": [
                "age",
                "monthly_income",
                "monthly_bill",
                "internet_usage_gb",
                "call_minutes",
                "support_tickets",
            ],
            "categorical": [
                "gender",
                "city",
                "education_level",
                "employment_status",
                "contract_type",
            ],
            "date": ["signup_date"],
            "text": ["customer_feedback"],
            "derived": {
                "tenure_column": "customer_tenure_days",
                "support_ticket_rule_column": "many_support_tickets_flag",
                "feedback_length_column": "customer_feedback_length",
                "feedback_keyword_prefix": "customer_feedback_keyword",
            },
            "excluded_from_training": ["customer_id", "churn"],
        },
        "baseline": {"support_tickets_threshold": 3},
    }


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "signup_date": ["2024-01-01", "2024-01-02", "2024-01-05"],
            "age": [24, 48, 36],
            "gender": ["Male", "Female", "Female"],
            "city": ["Lahore", "Lahore", "Karachi"],
            "education_level": ["Bachelor", "Master", "Bachelor"],
            "employment_status": ["Employed", "Employed", "Self-employed"],
            "monthly_income": [40000, 55000, 32000],
            "monthly_bill": [1200, 1500, 900],
            "internet_usage_gb": [12.0, 18.0, 7.0],
            "call_minutes": [100, 120, 80],
            "contract_type": ["Monthly", "6-Month", "Monthly"],
            "support_tickets": [1, 4, 3],
            "customer_feedback": [
                "Coverage is acceptable",
                "Billing issues keep happening",
                "Support response was slow",
            ],
            "churn": [0, 1, 1],
        }
    )


def test_tenure_uses_reference_date_when_provided() -> None:
    data = _sample_data()

    tenure = derive_customer_tenure_days(data, reference_date="2024-01-10")

    assert tenure.tolist() == [9, 8, 5]


def test_support_ticket_flag_marks_threshold_and_above() -> None:
    data = _sample_data()

    flags = derive_support_ticket_flag(data, threshold=3)

    assert flags.tolist() == [0, 1, 1]


def test_feedback_features_capture_keyword_signals() -> None:
    data = _sample_data()

    feedback_features = derive_customer_feedback_features(data)

    assert feedback_features["customer_feedback_keyword_billing"].tolist() == [0, 1, 0]
    assert feedback_features["customer_feedback_keyword_coverage"].tolist() == [1, 0, 0]
    assert feedback_features["customer_feedback_keyword_support"].tolist() == [0, 0, 1]
    assert feedback_features["customer_feedback_length"].tolist() == [22, 29, 25]


def test_add_derived_features_appends_expected_columns() -> None:
    data = _sample_data()
    config = _sample_config()

    engineered = add_derived_features(data, config, reference_date="2024-01-10")

    assert engineered["customer_tenure_days"].tolist() == [9, 8, 5]
    assert engineered["many_support_tickets_flag"].tolist() == [0, 1, 1]
    assert "customer_feedback_keyword_billing" in engineered.columns
    assert "customer_feedback_keyword_coverage" in engineered.columns
    assert "customer_feedback_keyword_support" in engineered.columns
