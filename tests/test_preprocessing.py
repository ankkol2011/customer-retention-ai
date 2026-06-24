"""Tests for the shared churn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import add_derived_features
from src.features.preprocessing import build_full_feature_pipeline, build_model_feature_schema


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
            "customer_id": [1, 2, 3, 4],
            "signup_date": ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-07"],
            "age": [24, 48, 36, 52],
            "gender": ["Male", "Female", "Female", "Male"],
            "city": ["Lahore", "Lahore", "Karachi", "Karachi"],
            "education_level": ["Bachelor", "Master", "Bachelor", "Master"],
            "employment_status": ["Employed", "Employed", "Self-employed", "Unemployed"],
            "monthly_income": [40000, 55000, 32000, 18000],
            "monthly_bill": [1200, 1500, 900, 2500],
            "internet_usage_gb": [12.0, 18.0, 7.0, 4.0],
            "call_minutes": [100, 120, 80, 30],
            "contract_type": ["Monthly", "6-Month", "Monthly", "Monthly"],
            "support_tickets": [1, 4, 3, 0],
            "customer_feedback": [
                "Coverage is acceptable",
                "Billing issues keep happening",
                "Support response was slow",
                "Service is fine",
            ],
            "churn": [0, 1, 1, 0],
        }
    )


def test_full_feature_pipeline_produces_expected_shape() -> None:
    data = _sample_data()
    config = _sample_config()
    pipeline = build_full_feature_pipeline(config, reference_date="2024-01-10")

    transformed = pipeline.fit_transform(data)
    schema = build_model_feature_schema(config)
    expected_feature_count = len(schema.numeric_for_preprocessing) + sum(
        data[column].nunique(dropna=True) for column in schema.categorical
    )

    assert transformed.shape == (4, expected_feature_count)


def test_full_feature_pipeline_handles_unseen_categories() -> None:
    data = _sample_data()
    config = _sample_config()
    pipeline = build_full_feature_pipeline(config, reference_date="2024-01-10")
    pipeline.fit(data)

    unseen = data.iloc[[0]].copy()
    unseen.loc[:, "city"] = "Faisalabad"

    transformed = pipeline.transform(unseen)
    seen_shape = pipeline.transform(data.iloc[[0]]).shape

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == seen_shape[1]
