"""Scoring service used by the Streamlit frontend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frontend.src.churn_ui.config import DEFAULT_PREDICTIONS_DIR, DEFAULT_REPORTS_DIR
from frontend.src.churn_ui.schemas import ScoringResult
from frontend.src.churn_ui.services.artifact_service import load_model_bundle
from src.models.inference import save_batch_predictions, score_batch_predictions


def score_uploaded_csv(
    data: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
    top_k: int = 100,
) -> ScoringResult:
    """Score an uploaded CSV and persist the outputs under the shared project outputs."""
    trained_model = load_model_bundle(model_path)
    scored, report = score_batch_predictions(data, trained_model, top_k=top_k)

    predictions_path = DEFAULT_PREDICTIONS_DIR / "frontend_churn_review_queue.csv"
    report_path = DEFAULT_REPORTS_DIR / "frontend_churn_review_queue.md"
    save_batch_predictions(scored, report, predictions_path, report_path)

    return ScoringResult(
        scored_frame=scored,
        report_markdown=report.to_markdown(),
        output_path=predictions_path,
    )
