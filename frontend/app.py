"""Streamlit app for telecom churn CSV scoring."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.src.churn_ui.pages.home import render_home_page
from frontend.src.churn_ui.pages.score_csv import render_score_csv_page
from frontend.src.churn_ui.pages.reports import render_reports_page


def main() -> None:
    st.set_page_config(
        page_title="Telecom Churn Review",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Telecom Churn Review")
    page = st.sidebar.radio(
        "Navigate",
        options=["Home", "Score CSV", "Reports"],
        index=0,
    )

    if page == "Home":
        render_home_page()
    elif page == "Score CSV":
        render_score_csv_page()
    else:
        render_reports_page()


if __name__ == "__main__":
    main()
