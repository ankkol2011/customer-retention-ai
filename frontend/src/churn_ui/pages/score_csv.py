"""CSV scoring page for the churn review frontend."""

from __future__ import annotations

import streamlit as st

from frontend.src.churn_ui.config import DEFAULT_MODEL_PATH
from frontend.src.churn_ui.services.scoring_service import score_uploaded_csv
from frontend.src.churn_ui.utils.csv_utils import read_uploaded_csv, validate_uploaded_csv


def render_score_csv_page() -> None:
    st.title("Score a CSV report")
    st.write("Upload a telecom customer CSV to generate a ranked churn review queue.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    top_k = st.number_input("Review queue size", min_value=1, value=100, step=1)

    if not uploaded_file:
        st.info("Upload a CSV to begin.")
        return

    try:
        data = read_uploaded_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.error(f"Could not read CSV: {exc}")
        return

    validation = validate_uploaded_csv(data, require_target=False)
    st.write(f"Rows: {validation.row_count} | Columns: {validation.column_count}")

    if not validation.passed:
        st.error("CSV validation failed.")
        for message in validation.messages:
            st.write(f"- {message}")
        return

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            "No trained model artifact found yet. Run the main MLOps training pipeline before scoring."
        )
        return

    if st.button("Score CSV"):
        try:
            result = score_uploaded_csv(data, model_path=DEFAULT_MODEL_PATH, top_k=int(top_k))
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error(f"Scoring failed: {exc}")
            return

        st.success("Scoring complete.")
        st.dataframe(result.scored_frame.head(20), use_container_width=True)
        st.markdown(result.report_markdown)
        if result.output_path is not None:
            st.caption(f"Saved scored CSV to: {result.output_path}")

