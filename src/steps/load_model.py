"""ZenML step wrapper for loading a trained churn model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from src.compat import zenml_step
from src.models.training import TrainedChurnModel


@zenml_step
def load_trained_model(
    model_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
) -> Annotated[TrainedChurnModel, "trained_model"]:
    """Load the persisted churn model bundle from disk."""
    return TrainedChurnModel.load(model_path)
