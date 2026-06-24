"""Tests for offline churn model evaluation."""

from __future__ import annotations

import pandas as pd

from src.models.evaluation import evaluate_trained_churn_model
from src.models.training import train_logistic_regression_model


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
        "evaluation": {
            "split": {
                "strategy": "stratified",
                "test_size": 0.25,
                "validation_size": 0.25,
                "random_seed": 42,
            }
        },
    }


def _sample_data() -> pd.DataFrame:
    rows = []
    base_rows = [
        {
            "signup_date": "2024-01-01",
            "age": 24,
            "gender": "Male",
            "city": "Lahore",
            "education_level": "Bachelor",
            "employment_status": "Employed",
            "monthly_income": 40000,
            "monthly_bill": 1200,
            "internet_usage_gb": 12.0,
            "call_minutes": 100,
            "contract_type": "Monthly",
            "support_tickets": 1,
            "customer_feedback": "Coverage is acceptable",
            "churn": 0,
        },
        {
            "signup_date": "2024-01-02",
            "age": 48,
            "gender": "Female",
            "city": "Karachi",
            "education_level": "Master",
            "employment_status": "Employed",
            "monthly_income": 55000,
            "monthly_bill": 1500,
            "internet_usage_gb": 18.0,
            "call_minutes": 120,
            "contract_type": "6-Month",
            "support_tickets": 4,
            "customer_feedback": "Billing issues keep happening",
            "churn": 1,
        },
        {
            "signup_date": "2024-01-03",
            "age": 36,
            "gender": "Female",
            "city": "Karachi",
            "education_level": "Bachelor",
            "employment_status": "Self-employed",
            "monthly_income": 32000,
            "monthly_bill": 900,
            "internet_usage_gb": 7.0,
            "call_minutes": 80,
            "contract_type": "Monthly",
            "support_tickets": 3,
            "customer_feedback": "Support response was slow",
            "churn": 1,
        },
        {
            "signup_date": "2024-01-04",
            "age": 52,
            "gender": "Male",
            "city": "Lahore",
            "education_level": "Master",
            "employment_status": "Unemployed",
            "monthly_income": 18000,
            "monthly_bill": 2500,
            "internet_usage_gb": 4.0,
            "call_minutes": 30,
            "contract_type": "Monthly",
            "support_tickets": 0,
            "customer_feedback": "Service is fine",
            "churn": 0,
        },
    ]

    for index in range(12):
        template = base_rows[index % len(base_rows)].copy()
        template["customer_id"] = 10000 + index
        template["signup_date"] = f"2024-01-{(index % 12) + 1:02d}"
        template["monthly_income"] = template["monthly_income"] + index * 500
        template["monthly_bill"] = template["monthly_bill"] + index * 50
        rows.append(template)

    return pd.DataFrame(rows)


def test_evaluation_report_compares_model_against_baseline() -> None:
    data = _sample_data()
    config = _sample_config()
    trained_model = train_logistic_regression_model(data, config)

    report = evaluate_trained_churn_model(data, trained_model, config)

    assert report.test_rows > 0
    assert "metrics" in report.model_metrics
    assert "metrics" in report.baseline_metrics
    assert "recall_delta" in report.metric_comparison
    assert report.slice_metrics
    assert "city" in report.slice_metrics
    assert "age_band" in report.slice_metrics
    assert "support_ticket_band" in report.slice_metrics


def test_evaluation_report_markdown_mentions_model_and_baseline() -> None:
    data = _sample_data()
    config = _sample_config()
    trained_model = train_logistic_regression_model(data, config)

    report = evaluate_trained_churn_model(data, trained_model, config)
    markdown = report.to_markdown()

    assert "# Model Evaluation Report" in markdown
    assert "## Model Metrics" in markdown
    assert "## Baseline Metrics" in markdown
    assert "## Slice Metrics" in markdown
