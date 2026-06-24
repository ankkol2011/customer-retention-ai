"""Home page for the churn review frontend."""

from __future__ import annotations

import streamlit as st

from frontend.src.churn_ui.config import DEFAULT_MODEL_PATH


def render_home_page() -> None:
    st.title("Telecom Churn Review")
    st.write(
        "Upload a customer CSV, score churn risk, and review the ranked queue before outreach."
    )

    st.subheader("Model artifact")
    st.code(str(DEFAULT_MODEL_PATH))

    if DEFAULT_MODEL_PATH.exists():
        st.success("Trained model artifact found.")
    else:
        st.warning("No trained model artifact found yet. Run the training pipeline first.")

