"""ZenML step wrapper for shared churn feature engineering."""

from __future__ import annotations

from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.features.feature_engineering import add_derived_features


@zenml_step
def engineer_feature_frame(
    data: pd.DataFrame,
    config_path: str = "configs/project.yaml",
) -> Annotated[pd.DataFrame, "engineered_features"]:
    """Apply the shared feature logic to a validated churn dataframe."""
    config = load_project_config(config_path)
    return add_derived_features(data, config)

