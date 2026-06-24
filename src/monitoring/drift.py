"""Lightweight drift monitoring for the telecom churn batch pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.inference import BatchPredictionReport
from src.models.training import TrainedChurnModel


@dataclass(frozen=True)
class DriftMetric:
    """Single feature drift measurement."""

    feature_name: str
    reference_mean: float | None
    current_mean: float | None
    mean_shift: float | None
    reference_missing_rate: float | None
    current_missing_rate: float | None
    missing_rate_shift: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "mean_shift": self.mean_shift,
            "reference_missing_rate": self.reference_missing_rate,
            "current_missing_rate": self.current_missing_rate,
            "missing_rate_shift": self.missing_rate_shift,
            "status": self.status,
        }


@dataclass(frozen=True)
class DriftReport:
    """Structured drift summary for a batch run."""

    model_name: str
    row_count: int
    feature_drift: dict[str, list[dict[str, Any]]]
    prediction_drift: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "row_count": self.row_count,
            "feature_drift": self.feature_drift,
            "prediction_drift": self.prediction_drift,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Drift Monitoring Report",
            "",
            "## Summary",
            f"- Model: {self.model_name}",
            f"- Rows monitored: {self.row_count}",
            "",
            "## Prediction Drift",
            f"- Current mean probability: {self.prediction_drift['current_mean_probability']:.3f}",
            f"- Reference mean probability: {self.prediction_drift['reference_mean_probability']:.3f}",
            f"- Mean shift: {self.prediction_drift['mean_shift']:.3f}",
            f"- Review queue size shift: {self.prediction_drift['review_queue_shift']}",
        ]

        if self.feature_drift:
            lines.extend(["", "## Feature Drift"])
            for group_name, entries in self.feature_drift.items():
                lines.append(f"### {group_name}")
                for entry in entries:
                    lines.append(
                        f"- {entry['feature_name']}: status={entry['status']}, "
                        f"reference_missing_rate={entry['reference_missing_rate'] if entry['reference_missing_rate'] is not None else 'n/a'}, "
                        f"current_missing_rate={entry['current_missing_rate'] if entry['current_missing_rate'] is not None else 'n/a'}, "
                        f"missing_rate_shift={entry['missing_rate_shift'] if entry['missing_rate_shift'] is not None else 'n/a'}"
                    )

        if self.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        lines.append("")
        return "\n".join(lines)


def monitor_batch_drift(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    batch_report: BatchPredictionReport,
    config: dict[str, Any],
) -> DriftReport:
    """Compare the current batch against the stored training reference."""
    monitoring_config = config["monitoring"]
    reference = trained_model.feature_summary.get("monitoring_reference", {})

    feature_drift = {
        "numeric": _build_numeric_drift(data, monitoring_config.get("key_numeric_features", []), reference),
        "categorical": _build_categorical_drift(
            data, monitoring_config.get("key_categorical_features", []), reference
        ),
    }
    prediction_drift = _build_prediction_drift(batch_report, trained_model)
    warnings = _build_warnings(feature_drift, prediction_drift)
    recommendations = _build_recommendations(feature_drift, prediction_drift)

    return DriftReport(
        model_name=trained_model.model_name,
        row_count=int(len(data)),
        feature_drift=feature_drift,
        prediction_drift=prediction_drift,
        warnings=warnings,
        recommendations=recommendations,
    )


def save_drift_report(report: DriftReport, output_path: str | Path) -> Path:
    """Persist the drift report as Markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path


def _build_numeric_drift(
    data: pd.DataFrame,
    features: list[str],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for feature_name in features:
        if feature_name not in data.columns:
            continue
        current_series = pd.to_numeric(data[feature_name], errors="coerce")
        current_mean = float(current_series.mean()) if current_series.notna().any() else None
        current_missing_rate = float(current_series.isna().mean()) if len(current_series) else None
        reference_entry = reference.get("numeric", {}).get(feature_name, {})
        reference_mean = reference_entry.get("mean")
        reference_missing_rate = reference_entry.get("missing_rate")
        mean_shift = _delta(current_mean, reference_mean)
        missing_rate_shift = _delta(current_missing_rate, reference_missing_rate)
        entries.append(
            DriftMetric(
                feature_name=feature_name,
                reference_mean=reference_mean,
                current_mean=current_mean,
                mean_shift=mean_shift,
                reference_missing_rate=reference_missing_rate,
                current_missing_rate=current_missing_rate,
                missing_rate_shift=missing_rate_shift,
                status=_drift_status(mean_shift=mean_shift, missing_rate_shift=missing_rate_shift),
            ).to_dict()
        )
    return entries


def _build_categorical_drift(
    data: pd.DataFrame,
    features: list[str],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for feature_name in features:
        if feature_name not in data.columns:
            continue
        current_series = data[feature_name].astype("string").fillna("UNKNOWN")
        current_missing_rate = float(current_series.isna().mean()) if len(current_series) else None
        current_distribution = current_series.value_counts(dropna=False, normalize=True).to_dict()
        reference_distribution = reference.get("categorical", {}).get(feature_name, {})
        distribution_shift = _distribution_shift(current_distribution, reference_distribution)
        entries.append(
            DriftMetric(
                feature_name=feature_name,
                reference_mean=None,
                current_mean=None,
                mean_shift=distribution_shift,
                reference_missing_rate=None,
                current_missing_rate=current_missing_rate,
                missing_rate_shift=current_missing_rate,
                status=_drift_status(mean_shift=distribution_shift, missing_rate_shift=current_missing_rate),
            ).to_dict()
        )
    return entries


def _build_prediction_drift(batch_report: BatchPredictionReport, trained_model: TrainedChurnModel) -> dict[str, Any]:
    reference_mean_probability = float(trained_model.validation_positive_rate)
    current_mean_probability = float(batch_report.summary["mean_probability"])
    review_queue_size = int(batch_report.review_queue_size)
    return {
        "reference_mean_probability": reference_mean_probability,
        "current_mean_probability": current_mean_probability,
        "mean_shift": current_mean_probability - reference_mean_probability,
        "review_queue_size": review_queue_size,
        "review_queue_shift": review_queue_size - int(trained_model.validation_rows * 0.5),
    }


def _build_warnings(feature_drift: dict[str, list[dict[str, Any]]], prediction_drift: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if abs(prediction_drift["mean_shift"]) > 0.15:
        warnings.append("Prediction mean shifted materially from the training reference.")
    if prediction_drift["review_queue_size"] == 0:
        warnings.append("Review queue is empty.")
    if any(entry["status"] == "significant" for entries in feature_drift.values() for entry in entries):
        warnings.append("At least one monitored feature shows significant drift.")
    return warnings


def _build_recommendations(
    feature_drift: dict[str, list[dict[str, Any]]],
    prediction_drift: dict[str, Any],
) -> list[str]:
    recommendations = [
        "Track the monitored features against the training baseline on every batch.",
        "Investigate upstream data changes before retraining when drift increases suddenly.",
    ]
    if any(entry["status"] == "significant" for entries in feature_drift.values() for entry in entries):
        recommendations.append("Review significant feature drift before promoting a new model.")
    if abs(prediction_drift["mean_shift"]) > 0.15:
        recommendations.append("Compare the current batch against recent stable batches to confirm the shift.")
    return recommendations


def _distribution_shift(current: dict[Any, float], reference: dict[Any, float]) -> float:
    keys = set(current) | set(reference)
    return float(sum(abs(current.get(key, 0.0) - reference.get(key, 0.0)) for key in keys) / 2.0)


def _delta(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return current - reference


def _drift_status(*, mean_shift: float | None, missing_rate_shift: float | None) -> str:
    if mean_shift is None and missing_rate_shift is None:
        return "unknown"
    mean_shift_value = abs(mean_shift) if mean_shift is not None else 0.0
    missing_shift_value = abs(missing_rate_shift) if missing_rate_shift is not None else 0.0
    if mean_shift_value > 0.5 or missing_shift_value > 0.1:
        return "significant"
    if mean_shift_value > 0.2 or missing_shift_value > 0.05:
        return "moderate"
    return "stable"


__all__ = [
    "DriftMetric",
    "DriftReport",
    "monitor_batch_drift",
    "save_drift_report",
]
