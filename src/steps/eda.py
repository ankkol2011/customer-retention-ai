"""EDA and feature understanding step for churn CSV reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.models.baselines import evaluate_support_ticket_rule
from src.validation.data_validation import load_csv_report


@dataclass(frozen=True)
class FeatureUnderstanding:
    """High-level feature usage guidance for the churn project."""

    include_for_training: list[str]
    exclude_from_training: list[str]
    derived_feature_opportunities: list[str]
    human_review_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAReport:
    """Structured EDA summary for a churn dataset."""

    row_count: int
    column_count: int
    target_distribution: dict[str, int]
    churn_rate: float | None
    data_overview: dict[str, Any]
    feature_understanding: FeatureUnderstanding
    numeric_summary: dict[str, dict[str, Any]]
    categorical_summary: dict[str, dict[str, Any]]
    date_summary: dict[str, dict[str, Any]]
    text_summary: dict[str, dict[str, Any]]
    baseline_summary: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target_distribution": self.target_distribution,
            "churn_rate": self.churn_rate,
            "data_overview": self.data_overview,
            "feature_understanding": self.feature_understanding.to_dict(),
            "numeric_summary": self.numeric_summary,
            "categorical_summary": self.categorical_summary,
            "date_summary": self.date_summary,
            "text_summary": self.text_summary,
            "baseline_summary": self.baseline_summary,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# EDA Report: Telecom Customer Churn",
            "",
            "## Data Overview",
            f"- Rows: {self.row_count}",
            f"- Columns: {self.column_count}",
            f"- Churn rate: {self.churn_rate:.2%}" if self.churn_rate is not None else "- Churn rate: unavailable",
        ]

        if self.data_overview:
            lines.append(f"- Dataset summary: {self.data_overview.get('summary', 'n/a')}")

        lines.extend(
            [
                "",
                "## Feature Understanding",
                f"- Include for training: {', '.join(self.feature_understanding.include_for_training)}",
                f"- Exclude from training: {', '.join(self.feature_understanding.exclude_from_training)}",
                f"- Derived opportunities: {', '.join(self.feature_understanding.derived_feature_opportunities)}",
            ]
        )

        if self.feature_understanding.human_review_notes:
            lines.extend(["", "## Human Review Notes"])
            lines.extend(f"- {note}" for note in self.feature_understanding.human_review_notes)

        if self.target_distribution:
            lines.extend(["", "## Target Distribution"])
            for label, count in sorted(self.target_distribution.items(), key=lambda item: item[0]):
                lines.append(f"- {label}: {count}")

        if self.baseline_summary:
            baseline_metrics = self.baseline_summary["metrics"]
            lines.extend(
                [
                    "",
                    "## Support-Ticket Baseline",
                    f"- Threshold: support_tickets >= {self.baseline_summary['threshold']}",
                    f"- Positive predictions: {self.baseline_summary['positive_predictions']}",
                    f"- Predicted positive rate: {self.baseline_summary['predicted_positive_rate']:.2%}",
                    f"- Precision: {baseline_metrics['precision']:.3f}",
                    f"- Recall: {baseline_metrics['recall']:.3f}",
                    f"- F1: {baseline_metrics['f1_score']:.3f}",
                ]
            )

        if self.numeric_summary:
            lines.extend(["", "## Numeric Feature Summary"])
            for column, summary in self.numeric_summary.items():
                mean_value = summary.get("mean")
                mean_text = f"{mean_value:.2f}" if mean_value is not None else "n/a"
                lines.append(
                    f"- {column}: min={summary.get('min')}, max={summary.get('max')}, mean={mean_text}"
                )

        if self.categorical_summary:
            lines.extend(["", "## Categorical Feature Summary"])
            for column, summary in self.categorical_summary.items():
                top_values = summary.get("top_values", [])
                top_text = ", ".join(f"{item['value']} ({item['count']})" for item in top_values)
                lines.append(
                    f"- {column}: {summary.get('unique_count')} unique values; top values: {top_text}"
                )

        if self.date_summary:
            lines.extend(["", "## Date Feature Summary"])
            for column, summary in self.date_summary.items():
                lines.append(
                    f"- {column}: min={summary.get('min_date')} max={summary.get('max_date')} "
                    f"span_days={summary.get('span_days')}"
                )

        if self.text_summary:
            lines.extend(["", "## Text Feature Summary"])
            for column, summary in self.text_summary.items():
                lines.append(
                    f"- {column}: avg_length={summary.get('average_length'):.1f}, "
                    f"unique_ratio={summary.get('unique_ratio'):.2f}"
                )

        if self.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        lines.append("")
        return "\n".join(lines)


@zenml_step
def generate_eda_report(
    config_path: str = "configs/project.yaml",
) -> Annotated[EDAReport, "eda_report"]:
    """Generate a structured EDA report for the configured churn dataset."""
    config = load_project_config(config_path)
    csv_path = config["data"]["training_csv"]
    data = load_csv_report(csv_path, config, require_target=True)
    return build_eda_report(data, config)


def build_eda_report(data: pd.DataFrame, config: dict[str, Any]) -> EDAReport:
    """Build the EDA summary used for feature understanding and baseline review."""
    data_config = config["data"]
    feature_config = config["features"]
    baseline_config = config["baseline"]

    target_column = data_config["target_column"]
    id_column = data_config["id_column"]

    target_distribution = _value_counts_as_ints(data[target_column]) if target_column in data.columns else {}
    churn_rate = (
        float(pd.to_numeric(data[target_column], errors="coerce").mean())
        if target_column in data.columns
        else None
    )

    feature_understanding = FeatureUnderstanding(
        include_for_training=_dedupe_preserve_order(
            *feature_config.get("numeric", []),
            *feature_config.get("categorical", []),
            *feature_config.get("date", []),
            *feature_config.get("text", []),
        ),
        exclude_from_training=_dedupe_preserve_order(
            *feature_config.get("excluded_from_training", []),
            id_column,
            target_column,
        ),
        derived_feature_opportunities=_dedupe_preserve_order(
            feature_config["derived"]["tenure_column"],
            feature_config["derived"]["support_ticket_rule_column"],
            feature_config["derived"]["feedback_length_column"],
            f"{feature_config['derived'].get('feedback_keyword_prefix', 'customer_feedback_keyword')}_billing",
            f"{feature_config['derived'].get('feedback_keyword_prefix', 'customer_feedback_keyword')}_coverage",
            f"{feature_config['derived'].get('feedback_keyword_prefix', 'customer_feedback_keyword')}_support",
        ),
        human_review_notes=[
            "Keep customer_id only for joins and output files; never use it as a feature.",
            "Convert signup_date into tenure features instead of using the raw date string.",
            "Treat customer_feedback as lightweight text signals in the MVP, not embeddings.",
            "Keep the support-ticket rule as the operational baseline until the model proves better.",
        ],
    )

    numeric_summary = _summarize_numeric_features(data, feature_config.get("numeric", []))
    categorical_summary = _summarize_categorical_features(data, feature_config.get("categorical", []))
    date_summary = _summarize_date_features(data, feature_config.get("date", []))
    text_summary = _summarize_text_features(data, feature_config.get("text", []))
    baseline_result = (
        evaluate_support_ticket_rule(
            data,
            target_column=target_column,
            support_tickets_column="support_tickets",
            threshold=int(baseline_config["support_tickets_threshold"]),
        ).to_dict()
        if target_column in data.columns and "support_tickets" in data.columns
        else None
    )

    warnings = _build_warnings(
        data,
        config,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        date_summary=date_summary,
        text_summary=text_summary,
    )

    recommendations = [
        "Use recall as the primary model-selection metric because false negatives are more costly.",
        "Keep a top-K human review queue rather than a fixed 0.5 probability threshold.",
        "Drop customer_id from the model matrix to avoid leakage.",
        "Derive tenure from signup_date before training.",
        "Use the support-ticket rule as the benchmark the first ML model must beat.",
    ]

    return EDAReport(
        row_count=int(len(data)),
        column_count=int(len(data.columns)),
        target_distribution=target_distribution,
        churn_rate=churn_rate,
        data_overview={
            "dataset_path": data_config["training_csv"],
            "id_column": id_column,
            "target_column": target_column,
            "positive_class": data_config.get("positive_class", 1),
            "summary": "CSV-first telecom churn dataset with labeled customer outcomes.",
        },
        feature_understanding=feature_understanding,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        date_summary=date_summary,
        text_summary=text_summary,
        baseline_summary=baseline_result,
        warnings=warnings,
        recommendations=recommendations,
    )


def save_eda_report(report: EDAReport, output_path: str | Path) -> Path:
    """Write the EDA report as Markdown for review."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_markdown(), encoding="utf-8")
    return output


def _value_counts_as_ints(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(index): int(value) for index, value in counts.items()}


def _dedupe_preserve_order(*items: str) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _summarize_numeric_features(
    data: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in data.columns:
            continue

        numeric_values = pd.to_numeric(data[column], errors="coerce")
        summary[column] = {
            "missing_count": int(numeric_values.isna().sum()),
            "min": float(numeric_values.min()) if numeric_values.notna().any() else None,
            "max": float(numeric_values.max()) if numeric_values.notna().any() else None,
            "mean": float(numeric_values.mean()) if numeric_values.notna().any() else None,
            "median": float(numeric_values.median()) if numeric_values.notna().any() else None,
            "std": float(numeric_values.std(ddof=0)) if numeric_values.notna().any() else None,
        }
    return summary


def _summarize_categorical_features(
    data: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in data.columns:
            continue

        values = data[column].astype("string")
        top_values = (
            values.value_counts(dropna=False)
            .head(3)
            .rename_axis("value")
            .reset_index(name="count")
        )
        summary[column] = {
            "unique_count": int(values.nunique(dropna=True)),
            "unique_ratio": float(values.nunique(dropna=True) / len(values)) if len(values) else 0.0,
            "top_values": [
                {"value": str(row["value"]), "count": int(row["count"])}
                for _, row in top_values.iterrows()
            ],
        }
    return summary


def _summarize_date_features(
    data: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in data.columns:
            continue

        parsed = pd.to_datetime(data[column], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            summary[column] = {
                "parseable": False,
                "min_date": None,
                "max_date": None,
                "span_days": None,
            }
            continue

        summary[column] = {
            "parseable": True,
            "min_date": valid.min().date().isoformat(),
            "max_date": valid.max().date().isoformat(),
            "span_days": int((valid.max() - valid.min()).days),
        }
    return summary


def _summarize_text_features(
    data: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in data.columns:
            continue

        values = data[column].fillna("").astype(str)
        lengths = values.str.len()
        summary[column] = {
            "average_length": float(lengths.mean()) if len(lengths) else 0.0,
            "median_length": float(lengths.median()) if len(lengths) else 0.0,
            "unique_ratio": float(values.nunique(dropna=False) / len(values)) if len(values) else 0.0,
        }
    return summary


def _build_warnings(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    numeric_summary: dict[str, dict[str, Any]],
    categorical_summary: dict[str, dict[str, Any]],
    date_summary: dict[str, dict[str, Any]],
    text_summary: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    data_config = config["data"]
    id_column = data_config["id_column"]
    target_column = data_config["target_column"]
    feature_config = config["features"]

    if id_column in data.columns:
        unique_ratio = data[id_column].nunique(dropna=True) / len(data) if len(data) else 0.0
        if unique_ratio >= 0.95:
            warnings.append(
                f"{id_column} is effectively unique per row and should stay out of the model matrix."
            )

    if target_column in data.columns:
        churn_rate = pd.to_numeric(data[target_column], errors="coerce").mean()
        if pd.notna(churn_rate) and (churn_rate < 0.2 or churn_rate > 0.8):
            warnings.append(
                f"Target balance is skewed at {churn_rate:.2%}; threshold tuning and PR metrics matter."
            )

    for column, summary in categorical_summary.items():
        unique_ratio = float(summary.get("unique_ratio", 0.0))
        if unique_ratio >= 0.25:
            warnings.append(
                f"Categorical feature {column} has high cardinality ({summary['unique_count']} unique values)."
            )

    for column, summary in text_summary.items():
        if float(summary.get("unique_ratio", 0.0)) >= 0.9:
            warnings.append(
                f"Text feature {column} is highly unique; treat it as lightweight signal features in the MVP."
            )

    for column, summary in date_summary.items():
        if column == "signup_date" and summary.get("parseable"):
            warnings.append(
                "signup_date is parseable; use it to derive tenure and consider time-aware validation if it reflects observation time."
            )

    for column, summary in numeric_summary.items():
        if column in feature_config.get("numeric", []) and summary.get("min") is not None:
            if float(summary["min"]) < 0:
                warnings.append(f"Numeric feature {column} has negative values; confirm they are valid.")

    return warnings
