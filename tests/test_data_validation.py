"""Tests for churn CSV validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.validation.data_validation import DataValidationError, validate_churn_dataframe


@pytest.fixture
def project_config() -> dict:
    return {
        "data": {
            "id_column": "customer_id",
            "target_column": "churn",
        },
        "validation": {
            "required_columns": [
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
                "churn",
            ],
            "allow_missing_values": False,
            "max_duplicate_customer_ids": 0,
            "numeric_ranges": {
                "age": {"min": 0, "max": 120},
                "monthly_income": {"min": 0},
                "monthly_bill": {"min": 0},
                "internet_usage_gb": {"min": 0},
                "call_minutes": {"min": 0},
                "support_tickets": {"min": 0},
            },
        },
    }


@pytest.fixture
def valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [10001, 10002],
            "signup_date": ["2019-01-01", "2019-01-02"],
            "age": [65, 22],
            "gender": ["Male", "Female"],
            "city": ["Multan", "Peshawar"],
            "education_level": ["Bachelor", "Master"],
            "employment_status": ["Employed", "Employed"],
            "monthly_income": [57643.0, 18207.0],
            "monthly_bill": [10049.0, 2752.0],
            "internet_usage_gb": [5.8, 11.9],
            "call_minutes": [52.0, 22.0],
            "contract_type": ["6-Month", "6-Month"],
            "support_tickets": [1, 2],
            "customer_feedback": [
                "Coverage is poor in my location",
                "Billing issues occurred multiple times",
            ],
            "churn": [0, 1],
        }
    )


def test_valid_training_data_passes(project_config: dict, valid_data: pd.DataFrame) -> None:
    result = validate_churn_dataframe(valid_data, project_config, require_target=True)

    assert result.passed is True
    assert result.errors == []
    assert result.row_count == 2


def test_missing_required_column_fails(project_config: dict, valid_data: pd.DataFrame) -> None:
    invalid_data = valid_data.drop(columns=["monthly_bill"])

    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_churn_dataframe(invalid_data, project_config)


def test_duplicate_customer_id_fails(project_config: dict, valid_data: pd.DataFrame) -> None:
    invalid_data = valid_data.copy()
    invalid_data.loc[1, "customer_id"] = invalid_data.loc[0, "customer_id"]

    with pytest.raises(DataValidationError, match="Duplicate customer_id"):
        validate_churn_dataframe(invalid_data, project_config)


def test_invalid_target_value_fails(project_config: dict, valid_data: pd.DataFrame) -> None:
    invalid_data = valid_data.copy()
    invalid_data.loc[1, "churn"] = 2

    with pytest.raises(DataValidationError, match="must contain only 0/1"):
        validate_churn_dataframe(invalid_data, project_config)


def test_negative_numeric_value_fails(project_config: dict, valid_data: pd.DataFrame) -> None:
    invalid_data = valid_data.copy()
    invalid_data.loc[0, "monthly_bill"] = -10

    with pytest.raises(DataValidationError, match="below minimum"):
        validate_churn_dataframe(invalid_data, project_config)


def test_scoring_data_can_omit_target(project_config: dict, valid_data: pd.DataFrame) -> None:
    scoring_data = valid_data.drop(columns=["churn"])

    result = validate_churn_dataframe(scoring_data, project_config, require_target=False)

    assert result.passed is True
    assert result.errors == []

