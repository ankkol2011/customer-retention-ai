"""Data validation for telecom churn CSV reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Structured result from a data validation run."""

    passed: bool
    errors: list[str]
    warnings: list[str]
    row_count: int
    column_count: int


class DataValidationError(ValueError):
    """Raised when a dataset fails validation and fail_fast is enabled."""


def validate_churn_dataframe(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_target: bool = True,
    fail_fast: bool = True,
) -> ValidationResult:
    """Validate a churn training or scoring dataframe.

    Training data requires the target column. Scoring data can omit the target
    column by setting ``require_target=False``.
    """
    data_config = config["data"]
    validation_config = config["validation"]
    target_column = data_config["target_column"]
    id_column = data_config["id_column"]

    required_columns = list(validation_config["required_columns"])
    if not require_target and target_column in required_columns:
        required_columns.remove(target_column)

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_validate_required_columns(data, required_columns))

    if validation_config.get("allow_missing_values") is False:
        errors.extend(_validate_missing_values(data))

    if id_column in data.columns:
        max_duplicates = int(validation_config.get("max_duplicate_customer_ids", 0))
        duplicate_count = int(data[id_column].duplicated().sum())
        if duplicate_count > max_duplicates:
            errors.append(
                f"Duplicate {id_column} values found: {duplicate_count}; "
                f"maximum allowed: {max_duplicates}"
            )

    if require_target and target_column in data.columns:
        errors.extend(_validate_binary_target(data[target_column], target_column))

    errors.extend(_validate_numeric_ranges(data, validation_config.get("numeric_ranges", {})))

    result = ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        row_count=len(data),
        column_count=len(data.columns),
    )

    if fail_fast and errors:
        raise DataValidationError("; ".join(errors))

    return result


def load_csv_report(csv_path: str, config: dict[str, Any], *, require_target: bool = True) -> pd.DataFrame:
    """Load and validate a churn CSV report."""
    data = pd.read_csv(csv_path)
    validate_churn_dataframe(data, config, require_target=require_target, fail_fast=True)
    return data


def _validate_required_columns(data: pd.DataFrame, required_columns: list[str]) -> list[str]:
    missing_columns = [column for column in required_columns if column not in data.columns]
    if not missing_columns:
        return []
    return [f"Missing required columns: {missing_columns}"]


def _validate_missing_values(data: pd.DataFrame) -> list[str]:
    missing_counts = data.isna().sum()
    columns_with_missing = {
        column: int(count) for column, count in missing_counts.items() if int(count) > 0
    }
    if not columns_with_missing:
        return []
    return [f"Missing values found: {columns_with_missing}"]


def _validate_binary_target(target: pd.Series, target_column: str) -> list[str]:
    non_null_target = target.dropna()
    if len(non_null_target) == 0:
        return []

    numeric_target = pd.to_numeric(non_null_target, errors="coerce")
    if numeric_target.isna().any():
        invalid_values = sorted(set(non_null_target[numeric_target.isna()].astype(str)))
        return [
            f"Target column {target_column} must contain only numeric 0/1 values; "
            f"invalid values: {invalid_values}"
        ]

    observed_values = set(numeric_target.unique())
    allowed_values = {0, 1, 0.0, 1.0}
    if observed_values.issubset(allowed_values):
        return []

    return [
        f"Target column {target_column} must contain only 0/1 values; "
        f"observed: {sorted(observed_values)}"
    ]


def _validate_numeric_ranges(
    data: pd.DataFrame,
    numeric_ranges: dict[str, dict[str, float]],
) -> list[str]:
    errors: list[str] = []

    for column, bounds in numeric_ranges.items():
        if column not in data.columns:
            continue

        numeric_values = pd.to_numeric(data[column], errors="coerce")
        invalid_numeric_count = int(numeric_values.isna().sum() - data[column].isna().sum())
        if invalid_numeric_count > 0:
            errors.append(f"Column {column} contains {invalid_numeric_count} non-numeric values")
            continue

        min_value = bounds.get("min")
        max_value = bounds.get("max")

        if min_value is not None:
            below_min_count = int((numeric_values < min_value).sum())
            if below_min_count:
                errors.append(
                    f"Column {column} has {below_min_count} values below minimum {min_value}"
                )

        if max_value is not None:
            above_max_count = int((numeric_values > max_value).sum())
            if above_max_count:
                errors.append(
                    f"Column {column} has {above_max_count} values above maximum {max_value}"
                )

    return errors
