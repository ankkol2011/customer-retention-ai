"""ZenML pipeline for drift monitoring on batch churn scoring."""

from __future__ import annotations

from pathlib import Path

from src.compat import zenml_pipeline
from src.config import load_project_config
from src.steps.load_data import load_scoring_data
from src.steps.load_model import load_trained_model
from src.steps.monitor_drift import monitor_drift
from src.steps.score_batch import score_batch


@zenml_pipeline
def churn_monitoring_pipeline(
    csv_path: str,
    config_path: str = "configs/project.yaml",
    model_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
    predictions_path: str | Path = "outputs/predictions/churn_review_queue.csv",
    scoring_report_path: str | Path = "outputs/reports/churn_review_queue.md",
    drift_report_path: str | Path = "outputs/reports/drift_report.md",
) -> None:
    """Compose scoring and drift monitoring into one ZenML pipeline."""
    config = load_project_config(config_path)
    scoring_data = load_scoring_data(csv_path=csv_path, config_path=config_path)
    trained_model = load_trained_model(model_path=model_path)
    batch_report = score_batch(
        scoring_data,
        trained_model,
        top_k=int(config["evaluation"]["review_queue"]["default_top_k"]),
        output_path=predictions_path,
        report_path=scoring_report_path,
    )
    monitor_drift(
        scoring_data,
        trained_model,
        batch_report,
        config_path=config_path,
        output_path=drift_report_path,
    )


def run_monitoring_pipeline(
    csv_path: str,
    config_path: str = "configs/project.yaml",
    model_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
    predictions_path: str | Path = "outputs/predictions/churn_review_queue.csv",
    scoring_report_path: str | Path = "outputs/reports/churn_review_queue.md",
    drift_report_path: str | Path = "outputs/reports/drift_report.md",
):
    """Execute the batch scoring plus drift monitoring pipeline."""
    return churn_monitoring_pipeline(
        csv_path=csv_path,
        config_path=config_path,
        model_path=model_path,
        predictions_path=predictions_path,
        scoring_report_path=scoring_report_path,
        drift_report_path=drift_report_path,
    )
