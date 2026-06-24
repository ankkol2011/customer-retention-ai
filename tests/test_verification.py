"""Tests for production verification helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.evaluation import evaluate_trained_churn_model, save_evaluation_report
from src.models.inference import save_batch_predictions, score_batch_predictions
from src.models.training import train_logistic_regression_model
from src.monitoring.drift import monitor_batch_drift, save_drift_report
from src.verification import verify_markdown_report, verify_model_artifact, verify_prediction_artifact, verify_production_handoff


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
        "monitoring": {
            "key_numeric_features": [
                "monthly_bill",
                "internet_usage_gb",
                "call_minutes",
                "support_tickets",
                "customer_tenure_days",
            ],
            "key_categorical_features": [
                "city",
                "contract_type",
                "employment_status",
            ],
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


def test_production_handoff_verification_passes(tmp_path: Path) -> None:
    data = _sample_data()
    config = _sample_config()
    trained_model = train_logistic_regression_model(data, config)

    model_path = tmp_path / "telecom_churn_model.joblib"
    trained_model.save(model_path)

    scored, batch_report = score_batch_predictions(data.drop(columns=["churn"]), trained_model, top_k=3)
    predictions_path = tmp_path / "churn_review_queue.csv"
    batch_report_path = tmp_path / "churn_review_queue.md"
    save_batch_predictions(scored, batch_report, predictions_path, batch_report_path)

    evaluation_report = evaluate_trained_churn_model(data, trained_model, config)
    evaluation_path = tmp_path / "model_evaluation.md"
    save_evaluation_report(evaluation_report, evaluation_path)

    drift_report = monitor_batch_drift(data.drop(columns=["churn"]), trained_model, batch_report, config)
    drift_path = tmp_path / "drift_report.md"
    save_drift_report(drift_report, drift_path)

    report = verify_production_handoff(
        model_path,
        predictions_path,
        evaluation_report_path=evaluation_path,
        drift_report_path=drift_path,
        expected_top_k=3,
    )

    assert report.passed
    assert len(report.checks) == 4
    assert "Production Verification Report" in report.to_markdown()


def test_prediction_verification_rejects_missing_columns(tmp_path: Path) -> None:
    predictions_path = tmp_path / "bad_predictions.csv"
    pd.DataFrame({"customer_id": [1, 2], "churn_probability": [0.8, 0.2]}).to_csv(predictions_path, index=False)

    check = verify_prediction_artifact(predictions_path)

    assert not check.passed
    assert "Missing required prediction columns" in check.messages[0]


def test_model_verification_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.joblib"

    check = verify_model_artifact(missing_path)

    assert not check.passed
    assert "does not exist" in check.messages[0]


def test_report_verification_checks_headings(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text("# Heading\n\nBody", encoding="utf-8")

    check = verify_markdown_report(
        report_path,
        name="sample_report",
        required_headings=["# Heading", "## Missing"],
    )

    assert not check.passed
    assert "Missing required headings" in check.messages[0]
