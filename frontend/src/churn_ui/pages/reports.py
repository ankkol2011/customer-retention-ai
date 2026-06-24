"""Reports page for the churn review frontend."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.src.churn_ui.config import DEFAULT_REPORTS_DIR


def _render_report(path: Path) -> None:
    if not path.exists():
        st.warning(f"Missing report: {path.name}")
        return
    st.subheader(path.name)
    st.text(path.read_text(encoding="utf-8"))


def render_reports_page() -> None:
    st.title("Generated Reports")
    st.write("These reports are produced by the main MLOps project and the frontend scoring flow.")

    for report_name in [
        "eda_report.md",
        "model_evaluation.md",
        "drift_report.md",
        "production_handoff.md",
        "production_readiness.md",
        "frontend_churn_review_queue.md",
    ]:
        _render_report(DEFAULT_REPORTS_DIR / report_name)

