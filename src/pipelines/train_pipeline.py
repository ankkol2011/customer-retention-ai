"""ZenML pipeline for training the telecom churn classifier."""

from __future__ import annotations

from pathlib import Path

from src.compat import zenml_pipeline
from src.steps.load_data import load_training_data
from src.steps.train_model import train_model


@zenml_pipeline
def churn_training_pipeline(
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
) -> None:
    """Compose the churn training steps into a ZenML pipeline."""
    training_data = load_training_data(config_path=config_path)
    train_model(training_data, config_path=config_path, output_path=output_path)


def run_training_pipeline(
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
):
    """Execute the ZenML training pipeline."""
    return churn_training_pipeline(config_path=config_path, output_path=output_path)

