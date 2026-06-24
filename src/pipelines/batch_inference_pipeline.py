"""ZenML pipeline for batch churn scoring."""

from __future__ import annotations

from pathlib import Path

from src.compat import zenml_pipeline
from src.config import load_project_config
from src.steps.load_data import load_scoring_data
from src.steps.load_model import load_trained_model
from src.steps.score_batch import score_batch


@zenml_pipeline
def churn_batch_inference_pipeline(
    csv_path: str,
    config_path: str = "configs/project.yaml",
    model_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
    output_path: str | Path = "outputs/predictions/churn_review_queue.csv",
    report_path: str | Path = "outputs/reports/churn_review_queue.md",
) -> None:
    """Compose the batch scoring workflow into a ZenML pipeline."""
    config = load_project_config(config_path)
    scoring_data = load_scoring_data(csv_path=csv_path, config_path=config_path)
    trained_model = load_trained_model(model_path=model_path)
    score_batch(
        scoring_data,
        trained_model,
        top_k=int(config["evaluation"]["review_queue"]["default_top_k"]),
        output_path=output_path,
        report_path=report_path,
    )


def run_batch_inference_pipeline(
    csv_path: str,
    config_path: str = "configs/project.yaml",
    model_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
    output_path: str | Path = "outputs/predictions/churn_review_queue.csv",
    report_path: str | Path = "outputs/reports/churn_review_queue.md",
):
    """Execute the ZenML batch scoring pipeline."""
    return churn_batch_inference_pipeline(
        csv_path=csv_path,
        config_path=config_path,
        model_path=model_path,
        output_path=output_path,
        report_path=report_path,
    )
