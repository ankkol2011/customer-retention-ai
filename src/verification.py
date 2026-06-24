"""Production verification helpers for the telecom churn pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.training import TrainedChurnModel


@dataclass(frozen=True)
class VerificationCheck:
    """Single verification gate for a production artifact."""

    name: str
    passed: bool
    messages: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "messages": self.messages,
            "details": self.details,
        }


@dataclass(frozen=True)
class VerificationReport:
    """Aggregate verification summary for a production handoff."""

    checks: list[VerificationCheck]
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "context": self.context,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Production Verification Report",
            "",
            "## Summary",
            f"- Passed: {self.passed}",
        ]

        if self.context:
            for key, value in self.context.items():
                lines.append(f"- {key}: {value}")

        lines.extend(["", "## Checks"])
        for check in self.checks:
            lines.append(f"### {check.name}")
            lines.append(f"- Passed: {check.passed}")
            if check.messages:
                lines.extend(f"- {message}" for message in check.messages)
            if check.details:
                for key, value in check.details.items():
                    lines.append(f"- {key}: {value}")

        lines.append("")
        return "\n".join(lines)


def verify_model_artifact(model_path: str | Path) -> VerificationCheck:
    """Verify that the trained model bundle exists and is loadable."""
    path = Path(model_path)
    if not path.exists():
        return VerificationCheck(
            name="model_artifact",
            passed=False,
            messages=[f"Model artifact does not exist: {path}"],
        )

    try:
        model = TrainedChurnModel.load(path)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return VerificationCheck(
            name="model_artifact",
            passed=False,
            messages=[f"Model artifact could not be loaded: {exc}"],
        )

    messages: list[str] = []
    details = {
        "model_name": model.model_name,
        "reference_date": model.reference_date,
        "train_rows": model.train_rows,
        "validation_rows": model.validation_rows,
        "test_rows": model.test_rows,
    }

    if not model.model_name:
        messages.append("Model name is missing.")
    if model.train_rows <= 0 or model.validation_rows <= 0 or model.test_rows <= 0:
        messages.append("Model split sizes must all be positive.")
    if not hasattr(model.pipeline, "predict_proba"):
        messages.append("Model pipeline does not expose predict_proba.")
    if "monitoring_reference" not in model.feature_summary:
        messages.append("Monitoring reference is missing from the model bundle.")

    return VerificationCheck(
        name="model_artifact",
        passed=not messages,
        messages=messages,
        details=details,
    )


def verify_prediction_artifact(
    predictions_path: str | Path,
    *,
    expected_top_k: int | None = None,
) -> VerificationCheck:
    """Verify the scored batch file produced by the inference pipeline."""
    path = Path(predictions_path)
    if not path.exists():
        return VerificationCheck(
            name="prediction_artifact",
            passed=False,
            messages=[f"Prediction artifact does not exist: {path}"],
        )

    data = pd.read_csv(path)
    required_columns = [
        "customer_id",
        "churn_probability",
        "predicted_label",
        "risk_band",
        "review_rank",
        "review_queue_flag",
        "review_priority",
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    messages: list[str] = []
    details = {
        "row_count": int(len(data)),
        "columns": list(data.columns),
    }

    if missing_columns:
        messages.append(f"Missing required prediction columns: {', '.join(missing_columns)}")
        return VerificationCheck(
            name="prediction_artifact",
            passed=False,
            messages=messages,
            details=details,
        )

    if data.empty:
        messages.append("Prediction file is empty.")

    probabilities = pd.to_numeric(data["churn_probability"], errors="coerce")
    if probabilities.isna().any():
        messages.append("Some churn probabilities are not numeric.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        messages.append("Churn probabilities must be in the [0, 1] range.")

    if not probabilities.is_monotonic_decreasing:
        messages.append("Prediction rows are not sorted by descending churn probability.")

    expected_ranks = list(range(1, len(data) + 1))
    actual_ranks = pd.to_numeric(data["review_rank"], errors="coerce").astype("Int64").tolist()
    if actual_ranks != expected_ranks:
        messages.append("Review ranks must be a contiguous 1..N sequence.")

    risk_bands = set(data["risk_band"].astype("string").fillna(""))
    allowed_risk_bands = {"high", "medium", "low"}
    invalid_risk_bands = sorted(risk_bands - allowed_risk_bands)
    if invalid_risk_bands:
        messages.append(f"Unexpected risk bands: {', '.join(invalid_risk_bands)}")

    queue_flags = data["review_queue_flag"].astype(bool)
    queue_size = int(queue_flags.sum())
    if expected_top_k is not None and queue_size != min(int(expected_top_k), len(data)):
        messages.append(
            f"Review queue size {queue_size} does not match expected top_k {expected_top_k}."
        )

    expected_queue_flags = pd.Series([rank <= queue_size for rank in expected_ranks], index=data.index)
    if not queue_flags.reset_index(drop=True).equals(expected_queue_flags.reset_index(drop=True)):
        messages.append("Review queue flags must be True for the ranked top queue only.")

    expected_priorities = queue_flags.map({True: "review", False: "monitor"})
    if not data["review_priority"].astype("string").reset_index(drop=True).equals(
        expected_priorities.astype("string").reset_index(drop=True)
    ):
        messages.append("Review priorities must match the review_queue_flag values.")

    details.update(
        {
            "queue_size": queue_size,
            "positive_predictions": int(pd.to_numeric(data["predicted_label"], errors="coerce").fillna(0).sum()),
            "predicted_positive_rate": float(pd.to_numeric(data["predicted_label"], errors="coerce").fillna(0).mean())
            if len(data)
            else 0.0,
        }
    )

    return VerificationCheck(
        name="prediction_artifact",
        passed=not messages,
        messages=messages,
        details=details,
    )


def verify_markdown_report(
    report_path: str | Path,
    *,
    name: str,
    required_headings: list[str] | None = None,
) -> VerificationCheck:
    """Verify that a Markdown report exists and contains expected headings."""
    path = Path(report_path)
    if not path.exists():
        return VerificationCheck(
            name=name,
            passed=False,
            messages=[f"Report does not exist: {path}"],
        )

    text = path.read_text(encoding="utf-8")
    messages: list[str] = []
    if not text.strip():
        messages.append("Report is empty.")

    missing_headings = [
        heading for heading in (required_headings or []) if heading not in text
    ]
    if missing_headings:
        messages.append(f"Missing required headings: {', '.join(missing_headings)}")

    return VerificationCheck(
        name=name,
        passed=not messages,
        messages=messages,
        details={
            "path": str(path),
            "line_count": len(text.splitlines()),
        },
    )


def verify_production_handoff(
    model_path: str | Path,
    predictions_path: str | Path,
    *,
    evaluation_report_path: str | Path | None = None,
    drift_report_path: str | Path | None = None,
    expected_top_k: int | None = None,
) -> VerificationReport:
    """Run the production handoff checks across the saved artifacts."""
    checks = [
        verify_model_artifact(model_path),
        verify_prediction_artifact(predictions_path, expected_top_k=expected_top_k),
        verify_markdown_report(
            evaluation_report_path or "",
            name="evaluation_report",
            required_headings=["# Model Evaluation Report", "## Model Metrics", "## Baseline Metrics"],
        )
        if evaluation_report_path is not None
        else VerificationCheck(
            name="evaluation_report",
            passed=True,
            messages=["Evaluation report check was skipped."],
        ),
        verify_markdown_report(
            drift_report_path or "",
            name="drift_report",
            required_headings=["# Drift Monitoring Report", "## Prediction Drift"],
        )
        if drift_report_path is not None
        else VerificationCheck(
            name="drift_report",
            passed=True,
            messages=["Drift report check was skipped."],
        ),
    ]

    context = {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "expected_top_k": expected_top_k,
    }
    if evaluation_report_path is not None:
        context["evaluation_report_path"] = str(evaluation_report_path)
    if drift_report_path is not None:
        context["drift_report_path"] = str(drift_report_path)

    return VerificationReport(checks=checks, context=context)


__all__ = [
    "VerificationCheck",
    "VerificationReport",
    "verify_markdown_report",
    "verify_model_artifact",
    "verify_prediction_artifact",
    "verify_production_handoff",
]
