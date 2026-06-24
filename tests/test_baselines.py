"""Tests for the support-ticket churn baseline and EDA report."""

from __future__ import annotations

import pandas as pd

from src.models.baselines import apply_support_ticket_rule, evaluate_support_ticket_rule
from src.steps.eda import build_eda_report


def _sample_config() -> dict:
    return {
        "data": {
            "id_column": "customer_id",
            "target_column": "churn",
            "training_csv": "data/telecom_customer_churn_feature_engineering.csv",
            "positive_class": 1,
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
            },
            "excluded_from_training": ["customer_id", "churn"],
        },
        "baseline": {"support_tickets_threshold": 3},
    }


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
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


def test_support_ticket_rule_marks_threshold_and_above() -> None:
    data = _sample_data()

    predictions = apply_support_ticket_rule(data, threshold=3)

    assert predictions.tolist() == [0, 1, 1, 0]


def test_support_ticket_rule_metrics_are_correct() -> None:
    data = _sample_data()

    result = evaluate_support_ticket_rule(data, threshold=3)

    assert result.threshold == 3
    assert result.positive_predictions == 2
    assert result.metrics.true_positives == 2
    assert result.metrics.true_negatives == 2
    assert result.metrics.false_positives == 0
    assert result.metrics.false_negatives == 0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0


def test_eda_report_captures_feature_understanding_and_warnings() -> None:
    data = _sample_data()
    config = _sample_config()

    report = build_eda_report(data, config)

    assert report.row_count == 4
    assert report.column_count == 15
    assert report.target_distribution == {"0": 2, "1": 2}
    assert report.baseline_summary["positive_predictions"] == 2
    assert "customer_id is effectively unique per row" in " ".join(report.warnings)
    assert "signup_date is parseable" in " ".join(report.warnings)
    assert "customer_tenure_days" in report.feature_understanding.derived_feature_opportunities
    assert "customer_id" in report.feature_understanding.exclude_from_training
