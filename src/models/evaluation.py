"""Offline evaluation utilities for the telecom churn classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.models.baselines import ClassificationMetrics, SupportTicketBaselineResult, evaluate_support_ticket_rule
from src.models.training import DatasetSplit, TrainedChurnModel, split_churn_dataset


@dataclass(frozen=True)
class SliceMetric:
    """Metrics computed for one business slice."""

    slice_name: str
    slice_value: str
    row_count: int
    positive_rate: float
    metrics: ClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "slice_value": self.slice_value,
            "row_count": self.row_count,
            "positive_rate": self.positive_rate,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Structured offline evaluation report for a trained churn model."""

    model_name: str
    target_column: str
    split_strategy: str
    random_seed: int
    test_rows: int
    model_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    metric_comparison: dict[str, float]
    slice_metrics: dict[str, list[dict[str, Any]]]
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "split_strategy": self.split_strategy,
            "random_seed": self.random_seed,
            "test_rows": self.test_rows,
            "model_metrics": self.model_metrics,
            "baseline_metrics": self.baseline_metrics,
            "metric_comparison": self.metric_comparison,
            "slice_metrics": self.slice_metrics,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Model Evaluation Report: Telecom Customer Churn",
            "",
            "## Overall Evaluation",
            f"- Model: {self.model_name}",
            f"- Target: {self.target_column}",
            f"- Test rows: {self.test_rows}",
            f"- Split strategy: {self.split_strategy}",
            f"- Random seed: {self.random_seed}",
            "",
            "## Model Metrics",
            f"- Accuracy: {self.model_metrics['metrics']['accuracy']:.3f}",
            f"- Precision: {self.model_metrics['metrics']['precision']:.3f}",
            f"- Recall: {self.model_metrics['metrics']['recall']:.3f}",
            f"- F1: {self.model_metrics['metrics']['f1_score']:.3f}",
            f"- PR AUC: {self.model_metrics['pr_auc']:.3f}",
            f"- Brier score: {self.model_metrics['brier_score']:.3f}",
            f"- Positive predictions: {self.model_metrics['positive_predictions']}",
            "",
            "## Baseline Metrics",
            f"- Threshold: support_tickets >= {self.baseline_metrics['threshold']}",
            f"- Precision: {self.baseline_metrics['metrics']['precision']:.3f}",
            f"- Recall: {self.baseline_metrics['metrics']['recall']:.3f}",
            f"- F1: {self.baseline_metrics['metrics']['f1_score']:.3f}",
            f"- Positive predictions: {self.baseline_metrics['positive_predictions']}",
            "",
            "## Metric Comparison",
            f"- Recall delta: {self.metric_comparison['recall_delta']:.3f}",
            f"- Precision delta: {self.metric_comparison['precision_delta']:.3f}",
            f"- F1 delta: {self.metric_comparison['f1_delta']:.3f}",
        ]

        if self.slice_metrics:
            lines.extend(["", "## Slice Metrics"])
            for slice_name, slice_entries in self.slice_metrics.items():
                lines.append(f"### {slice_name}")
                for entry in slice_entries:
                    metrics = entry["metrics"]
                    lines.append(
                        f"- {entry['slice_value']}: rows={entry['row_count']}, "
                        f"positive_rate={entry['positive_rate']:.3f}, "
                        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
                        f"f1={metrics['f1_score']:.3f}"
                    )

        if self.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        lines.append("")
        return "\n".join(lines)


def evaluate_trained_churn_model(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    config: dict[str, Any],
) -> EvaluationReport:
    """Evaluate a fitted churn model against the held-out test split and baseline."""
    split = split_churn_dataset(data, config)
    target_column = config["data"]["target_column"]
    baseline_config = config["baseline"]
    evaluation_config = config["evaluation"]

    test_features = split.test.drop(columns=[target_column])
    y_true = pd.to_numeric(split.test[target_column], errors="coerce").astype(int)
    y_pred = trained_model.predict(test_features)
    y_prob = trained_model.predict_proba(test_features)

    model_metrics = _evaluate_predictions(y_true, y_pred, y_prob)
    baseline_result = evaluate_support_ticket_rule(
        split.test,
        target_column=target_column,
        support_tickets_column="support_tickets",
        threshold=int(baseline_config["support_tickets_threshold"]),
    )

    slice_metrics = _build_slice_metrics(
        split.test,
        y_true=y_true,
        y_pred=y_pred,
        slice_columns=list(config["features"].get("categorical", [])) + ["age", "support_tickets"],
    )

    comparison = {
        "recall_delta": model_metrics["metrics"]["recall"] - baseline_result.metrics.recall,
        "precision_delta": model_metrics["metrics"]["precision"] - baseline_result.metrics.precision,
        "f1_delta": model_metrics["metrics"]["f1_score"] - baseline_result.metrics.f1_score,
    }

    warnings = _build_warnings(split, model_metrics, baseline_result, slice_metrics)
    recommendations = _build_recommendations(model_metrics, baseline_result, comparison)

    return EvaluationReport(
        model_name=trained_model.model_name,
        target_column=target_column,
        split_strategy=str(evaluation_config["split"]["strategy"]),
        random_seed=int(evaluation_config["split"]["random_seed"]),
        test_rows=int(len(split.test)),
        model_metrics=model_metrics,
        baseline_metrics=baseline_result.to_dict(),
        metric_comparison=comparison,
        slice_metrics=slice_metrics,
        warnings=warnings,
        recommendations=recommendations,
    )


def save_evaluation_report(report: EvaluationReport, output_path: str | Path) -> Path:
    """Persist the evaluation report as Markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path


def _evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, Any]:
    metrics = _binary_classification_metrics(y_true, y_pred)
    positive_predictions = int(y_pred.sum())
    positive_rate = positive_predictions / len(y_pred) if len(y_pred) else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(y_true) else 0.0
    brier = float(brier_score_loss(y_true, y_prob)) if len(y_true) else 0.0
    return {
        "metrics": metrics.to_dict(),
        "positive_predictions": positive_predictions,
        "predicted_positive_rate": positive_rate,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }


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


def _build_slice_metrics(
    data: pd.DataFrame,
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    slice_columns: list[str],
) -> dict[str, list[dict[str, Any]]]:
    slice_metrics: dict[str, list[dict[str, Any]]] = {}
    for column in slice_columns:
        if column not in data.columns:
            continue

        if column == "age":
            series = _age_band(data[column])
            slice_name = "age_band"
        elif column == "support_tickets":
            series = _support_ticket_band(data[column])
            slice_name = "support_ticket_band"
        else:
            series = data[column].astype("string").fillna("UNKNOWN")
            slice_name = column

        entries: list[dict[str, Any]] = []
        for slice_value in sorted(series.dropna().unique(), key=str):
            mask = series == slice_value
            if not mask.any():
                continue
            slice_y_true = y_true[mask]
            slice_y_pred = y_pred[mask]
            metrics = _binary_classification_metrics(slice_y_true, slice_y_pred)
            entries.append(
                SliceMetric(
                    slice_name=slice_name,
                    slice_value=str(slice_value),
                    row_count=int(mask.sum()),
                    positive_rate=float(slice_y_true.mean()) if len(slice_y_true) else 0.0,
                    metrics=metrics,
                ).to_dict()
            )
        if entries:
            slice_metrics[slice_name] = entries
    return slice_metrics


def _build_warnings(
    split: DatasetSplit,
    model_metrics: dict[str, Any],
    baseline_result: SupportTicketBaselineResult,
    slice_metrics: dict[str, list[dict[str, Any]]],
) -> list[str]:
    warnings: list[str] = []
    if split.test.empty:
        warnings.append("Test split is empty; evaluation is not meaningful.")
    if baseline_result.metrics.recall >= model_metrics["metrics"]["recall"]:
        warnings.append("Model recall does not beat the support-ticket baseline.")
    if model_metrics["predicted_positive_rate"] > 0.5:
        warnings.append("Model flags more than half the test set for review.")
    if not slice_metrics:
        warnings.append("No slice metrics were produced.")
    return warnings


def _build_recommendations(
    model_metrics: dict[str, Any],
    baseline_result: SupportTicketBaselineResult,
    comparison: dict[str, float],
) -> list[str]:
    recommendations = [
        "Use recall as the primary gate because false negatives are the costly error.",
        "Review the top-K retention queue rather than a fixed 0.5 cutoff.",
    ]
    if comparison["recall_delta"] > 0:
        recommendations.append("The model improves recall over the support-ticket rule.")
    else:
        recommendations.append("The model does not improve recall over the support-ticket rule yet.")

    if comparison["precision_delta"] < 0:
        recommendations.append("Precision fell relative to the baseline; check review queue quality.")
    return recommendations


def _age_band(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[0, 24, 34, 44, 54, 120],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
        include_lowest=True,
        right=True,
    ).astype("string").fillna("UNKNOWN")


def _support_ticket_band(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"],
        include_lowest=True,
        right=True,
    ).astype("string").fillna("UNKNOWN")


__all__ = [
    "EvaluationReport",
    "SliceMetric",
    "evaluate_trained_churn_model",
    "save_evaluation_report",
]
