# Churn Review Frontend

Simple Streamlit app for uploading a CSV and scoring telecom churn risk.

## What it does

- Upload a CSV report.
- Validate the upload against the project schema.
- Load the trained churn model bundle from the main MLOps project.
- Score the CSV and show the ranked review queue.
- Download the scored results as CSV.

## Why this is separate

This frontend is intentionally isolated from the main MLOps pipeline code.
That keeps the UI replaceable later with React or Vue while preserving a small
Python service layer for scoring.

## Development

Install the frontend dependencies, then run Streamlit:

```bash
cd frontend
python -m pip install -e .
streamlit run app.py
```

The app expects a trained model artifact at:

```text
outputs/models/telecom_churn_classifier.joblib
```

If the artifact is missing, the UI shows a clear message instead of crashing.
