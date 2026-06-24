"""ZenML step wrapper for training the churn classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.models.training import TrainedChurnModel, train_logistic_regression_model


@zenml_step
def train_model(
    data: pd.DataFrame,
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/models/telecom_churn_classifier.joblib",
) -> Annotated[TrainedChurnModel, "trained_model"]:
    """Train the first churn model on validated data."""
    config = load_project_config(config_path)
    trained_model = train_logistic_regression_model(data, config)
    trained_model.save(output_path)
    return trained_model
