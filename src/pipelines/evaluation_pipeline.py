"""ZenML pipeline for offline model evaluation."""

from __future__ import annotations

from pathlib import Path

from src.compat import zenml_pipeline
from src.steps.evaluate_model import evaluate_model
from src.steps.load_data import load_training_data
from src.steps.train_model import train_model


@zenml_pipeline
def churn_evaluation_pipeline(
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/reports/model_evaluation.md",
) -> None:
    """Compose loading, training, and evaluation into a ZenML pipeline."""
    training_data = load_training_data(config_path=config_path)
    trained_model = train_model(training_data, config_path=config_path)
    evaluate_model(training_data, trained_model, config_path=config_path, output_path=output_path)


def run_evaluation_pipeline(
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/reports/model_evaluation.md",
):
    """Execute the ZenML offline evaluation pipeline."""
    return churn_evaluation_pipeline(config_path=config_path, output_path=output_path)

