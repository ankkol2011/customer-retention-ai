"""Tests for batch churn scoring and deployment wiring."""

from __future__ import annotations

import pandas as pd

from src.models.inference import score_batch_predictions
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
            },
            "review_queue": {
                "default_top_k": 3,
            },
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

    for index in range(8):
        template = base_rows[index % len(base_rows)].copy()
        template["customer_id"] = 10000 + index
        template["signup_date"] = f"2024-01-{(index % 8) + 1:02d}"
        template["monthly_income"] = template["monthly_income"] + index * 500
        template["monthly_bill"] = template["monthly_bill"] + index * 50
        rows.append(template)

    return pd.DataFrame(rows)


def test_batch_scoring_adds_rank_and_risk_bands() -> None:
    data = _sample_data()
    config = _sample_config()
    trained_model = train_logistic_regression_model(data, config)

    scored, report = score_batch_predictions(data.drop(columns=["churn"]), trained_model, top_k=3)

    assert len(scored) == len(data)
    assert "churn_probability" in scored.columns
    assert "review_rank" in scored.columns
    assert "risk_band" in scored.columns
    assert report.review_queue_size == 3
    assert report.row_count == len(data)
    assert scored["review_rank"].min() == 1
    assert scored["review_rank"].max() == len(data)


def test_batch_scoring_report_mentions_queue_size() -> None:
    data = _sample_data()
    config = _sample_config()
    trained_model = train_logistic_regression_model(data, config)

    _, report = score_batch_predictions(data.drop(columns=["churn"]), trained_model, top_k=3)
    markdown = report.to_markdown()

    assert "# Batch Churn Scoring Report" in markdown
    assert "Review queue size" in markdown
    assert "Risk Bands" in markdown
