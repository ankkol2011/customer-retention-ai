"""ZenML step wrapper for batch churn scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

from src.compat import zenml_step
from src.models.inference import BatchPredictionReport, save_batch_predictions, score_batch_predictions
from src.models.training import TrainedChurnModel


@zenml_step
def score_batch(
    data: pd.DataFrame,
    trained_model: TrainedChurnModel,
    *,
    top_k: int = 100,
    output_path: str | Path = "outputs/predictions/churn_review_queue.csv",
    report_path: str | Path = "outputs/reports/churn_review_queue.md",
) -> Annotated[BatchPredictionReport, "batch_prediction_report"]:
    """Score a batch of customers and persist the ranked review queue."""
    scored, report = score_batch_predictions(data, trained_model, top_k=top_k)
    save_batch_predictions(scored, report, output_path, report_path)
    return report
