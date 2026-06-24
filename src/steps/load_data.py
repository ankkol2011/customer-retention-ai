"""ZenML data loading step for churn CSV reports."""

from __future__ import annotations

from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.validation.data_validation import load_csv_report


@zenml_step
def load_training_data(
    config_path: str = "configs/project.yaml",
) -> Annotated[pd.DataFrame, "training_data"]:
    """Load and validate the configured training CSV."""
    config = load_project_config(config_path)
    csv_path = config["data"]["training_csv"]
    return load_csv_report(csv_path, config, require_target=True)


@zenml_step
def load_scoring_data(
    csv_path: str,
    config_path: str = "configs/project.yaml",
) -> Annotated[pd.DataFrame, "scoring_data"]:
    """Load and validate a scoring CSV that may not contain the target column."""
    config = load_project_config(config_path)
    return load_csv_report(csv_path, config, require_target=False)
