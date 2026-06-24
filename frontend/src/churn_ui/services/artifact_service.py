"""Artifact loading helpers for the frontend."""

from __future__ import annotations

from pathlib import Path

from frontend.src.churn_ui.config import DEFAULT_MODEL_PATH
from src.models.training import TrainedChurnModel


def get_model_path() -> Path:
    """Return the expected model bundle path."""
    return DEFAULT_MODEL_PATH


def load_model_bundle(model_path: str | Path | None = None) -> TrainedChurnModel:
    """Load the trained churn model if it exists."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train the model in the main MLOps project first."
        )
    return TrainedChurnModel.load(path)

