"""ZenML step wrapper for offline model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.models.evaluation import EvaluationReport, evaluate_trained_churn_model, save_evaluation_report
from src.models.training import TrainedChurnModel


@zenml_step
def evaluate_model(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/reports/model_evaluation.md",
) -> Annotated[EvaluationReport, "evaluation_report"]:
    """Evaluate a trained churn model and save the Markdown report."""
    config = load_project_config(config_path)
    report = evaluate_trained_churn_model(data, trained_model, config)
    save_evaluation_report(report, output_path)
    return report

