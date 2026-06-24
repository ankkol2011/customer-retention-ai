"""Batch inference utilities for the telecom churn classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.training import TrainedChurnModel


HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.50


@dataclass(frozen=True)
class BatchPredictionReport:
    """Structured output for a batch churn scoring run."""

    model_name: str
    row_count: int
    review_queue_size: int
    positive_predictions: int
    predicted_positive_rate: float
    risk_band_counts: dict[str, int]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "row_count": self.row_count,
            "review_queue_size": self.review_queue_size,
            "positive_predictions": self.positive_predictions,
            "predicted_positive_rate": self.predicted_positive_rate,
            "risk_band_counts": self.risk_band_counts,
            "summary": self.summary,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Batch Churn Scoring Report",
            "",
            "## Summary",
            f"- Model: {self.model_name}",
            f"- Rows scored: {self.row_count}",
            f"- Review queue size: {self.review_queue_size}",
            f"- Positive predictions: {self.positive_predictions}",
            f"- Predicted positive rate: {self.predicted_positive_rate:.2%}",
            "",
            "## Risk Bands",
        ]
        for band, count in sorted(self.risk_band_counts.items(), key=lambda item: item[0]):
            lines.append(f"- {band}: {count}")

        if self.summary:
            lines.extend(["", "## Summary Metrics"])
            for key, value in self.summary.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.3f}")
                else:
                    lines.append(f"- {key}: {value}")

        if self.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        lines.append("")
        return "\n".join(lines)


def score_batch_predictions(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    *,
    top_k: int = 100,
) -> tuple[pd.DataFrame, BatchPredictionReport]:
    """Score a churn batch and return the ranked retention queue plus summary."""
    scored = data.copy()
    probabilities = trained_model.predict_proba(scored)
    predictions = trained_model.predict(scored)

    scored["churn_probability"] = probabilities
    scored["predicted_label"] = predictions
    scored["risk_band"] = scored["churn_probability"].apply(_risk_band)
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    scored["review_rank"] = scored.index + 1
    scored["review_queue_flag"] = scored["review_rank"] <= int(top_k)
    scored["review_priority"] = scored["review_queue_flag"].map({True: "review", False: "monitor"})

    risk_band_counts = {
        band: int((scored["risk_band"] == band).sum())
        for band in ["high", "medium", "low"]
    }
    review_queue_size = int(scored["review_queue_flag"].sum())
    positive_predictions = int(scored["predicted_label"].sum())
    predicted_positive_rate = positive_predictions / len(scored) if len(scored) else 0.0

    report = BatchPredictionReport(
        model_name=trained_model.model_name,
        row_count=int(len(scored)),
        review_queue_size=review_queue_size,
        positive_predictions=positive_predictions,
        predicted_positive_rate=predicted_positive_rate,
        risk_band_counts=risk_band_counts,
        summary={
            "top_k": int(top_k),
            "max_probability": float(scored["churn_probability"].max()) if len(scored) else 0.0,
            "mean_probability": float(scored["churn_probability"].mean()) if len(scored) else 0.0,
        },
        warnings=_build_warnings(scored, top_k=top_k),
        recommendations=_build_recommendations(scored, review_queue_size=review_queue_size, top_k=top_k),
    )
    return scored, report


def save_batch_predictions(
    scored: pd.DataFrame,
    report: BatchPredictionReport,
    output_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    """Persist scored predictions and a Markdown summary."""
    csv_path = Path(output_path)
    md_path = Path(report_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(csv_path, index=False)
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return csv_path, md_path


def _risk_band(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _build_warnings(scored: pd.DataFrame, *, top_k: int) -> list[str]:
    warnings: list[str] = []
    if scored.empty:
        warnings.append("Scoring batch is empty.")
    if top_k <= 0:
        warnings.append("top_k must be positive.")
    if scored["churn_probability"].isna().any():
        warnings.append("Some churn probabilities are missing.")
    if scored["churn_probability"].nunique(dropna=True) <= 1:
        warnings.append("Scores are nearly constant; review feature pipeline and model fit.")
    return warnings


def _build_recommendations(
    scored: pd.DataFrame,
    *,
    review_queue_size: int,
    top_k: int,
) -> list[str]:
    recommendations = [
        "Use the ranked queue for human retention review.",
        "Keep the support-ticket rule as the fallback comparison point.",
    ]
    if review_queue_size < top_k:
        recommendations.append("The batch is smaller than the review capacity; review all high-risk cases.")
    else:
        recommendations.append("Use review_rank to prioritize outreach when the queue exceeds capacity.")
    return recommendations


__all__ = [
    "BatchPredictionReport",
    "MEDIUM_RISK_THRESHOLD",
    "HIGH_RISK_THRESHOLD",
    "save_batch_predictions",
    "score_batch_predictions",
]
