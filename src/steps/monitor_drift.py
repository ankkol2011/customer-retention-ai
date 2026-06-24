"""ZenML step wrapper for batch drift monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.config import load_project_config
from src.models.inference import BatchPredictionReport
from src.models.training import TrainedChurnModel
from src.monitoring.drift import DriftReport, monitor_batch_drift, save_drift_report


@zenml_step
def monitor_drift(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    batch_report: BatchPredictionReport,
    config_path: str = "configs/project.yaml",
    output_path: str | Path = "outputs/reports/drift_report.md",
) -> Annotated[DriftReport, "drift_report"]:
    """Generate and persist a lightweight drift report for the batch."""
    config = load_project_config(config_path)
    report = monitor_batch_drift(data, trained_model, batch_report, config)
    save_drift_report(report, output_path)
    return report
