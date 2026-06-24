"""Baseline churn rules and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary classification metrics for a churn baseline or model."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupportTicketBaselineResult:
    """Evaluation output for the support-ticket churn rule."""

    threshold: int
    positive_predictions: int
    predicted_positive_rate: float
    metrics: ClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "positive_predictions": self.positive_predictions,
            "predicted_positive_rate": self.predicted_positive_rate,
            "metrics": self.metrics.to_dict(),
        }


def apply_support_ticket_rule(
    data: pd.DataFrame,
    *,
    support_tickets_column: str = "support_tickets",
    threshold: int = 3,
) -> pd.Series:
    """Return 1 when support tickets meet or exceed the threshold."""
    if support_tickets_column not in data.columns:
        raise KeyError(f"Missing required column: {support_tickets_column}")

    support_tickets = pd.to_numeric(data[support_tickets_column], errors="coerce")
    if support_tickets.isna().any():
        raise ValueError(f"Column {support_tickets_column} contains non-numeric values")

    return (support_tickets >= threshold).astype(int)


def evaluate_support_ticket_rule(
    data: pd.DataFrame,
    *,
    target_column: str = "churn",
    support_tickets_column: str = "support_tickets",
    threshold: int = 3,
) -> SupportTicketBaselineResult:
    """Evaluate the current support-ticket outreach rule against churn labels."""
    if target_column not in data.columns:
        raise KeyError(f"Missing required target column: {target_column}")

    y_true = pd.to_numeric(data[target_column], errors="coerce")
    if y_true.isna().any():
        raise ValueError(f"Target column {target_column} contains non-numeric values")

    y_pred = apply_support_ticket_rule(
        data,
        support_tickets_column=support_tickets_column,
        threshold=threshold,
    )

    metrics = _binary_classification_metrics(y_true.astype(int), y_pred)
    positive_predictions = int(y_pred.sum())
    predicted_positive_rate = positive_predictions / len(data) if len(data) else 0.0

    return SupportTicketBaselineResult(
        threshold=threshold,
        positive_predictions=positive_predictions,
        predicted_positive_rate=predicted_positive_rate,
        metrics=metrics,
    )


def _binary_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    support = int((y_true == 1).sum())
    total = int(len(y_true))
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return ClassificationMetrics(
        true_negatives=true_negative,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_positives=true_positive,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        support=support,
    )
