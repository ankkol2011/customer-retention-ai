# Customer Retention AI

Customer Retention AI is a production-minded machine learning project for telecom churn prediction. It ranks customers by churn risk so a retention team can review the highest-risk cases and decide whether to intervene.

The model is advisory. It does not contact customers automatically or assign offers. Human review stays in the loop.

## What This Project Does

- Validates telecom CSV reports before they reach the model.
- Trains a churn classifier on structured customer data.
- Compares the model against a rule-based baseline.
- Scores new customer snapshots in batch.
- Produces a ranked retention review queue.
- Monitors batch drift and prediction shift over time.
- Exposes a simple Streamlit frontend for CSV upload and scoring.

## Core Features

- **Binary churn prediction** using `churn` as the target.
- **Recall-focused evaluation** because missing churners is more expensive than reviewing extra customers.
- **Baseline comparison** against the existing `support_tickets >= 3` rule.
- **Shared preprocessing** so training and scoring use the same feature logic.
- **Batch inference** for daily or weekly retention review lists.
- **Drift monitoring** for key numeric and categorical features.
- **Verification checks** for saved model bundles and generated reports.
- **Frontend upload flow** for quick testing without touching the pipeline code.

## Project Structure

```text
configs/           Project configuration and feature policy
data/              Input CSVs and local data folders
frontend/          Separate Streamlit app for CSV upload and scoring
outputs/           Generated reports, predictions, and artifacts
src/               Main MLOps code: validation, features, models, pipelines, monitoring
tests/             Unit tests for the pipeline and supporting logic
.github/           CI workflow for syntax and tests
```

## Architecture Summary

The project follows a standard tabular MLOps flow:

1. Data loading and validation
2. EDA and baseline analysis
3. Feature engineering and preprocessing
4. Model training
5. Offline evaluation
6. Batch scoring
7. Drift monitoring
8. Production verification and handoff

## Model and Data

- **Problem type:** binary classification
- **Target:** `churn`
- **Positive class:** `1`
- **Primary metric:** recall for churners
- **Baseline:** support-ticket rule with threshold `>= 3`
- **Initial dataset:** `data/telecom_customer_churn_feature_engineering.csv`

The main training data includes numeric, categorical, date, and text fields. `customer_id` is kept for joins and outputs, but it is not used as a predictive feature.

## Frontend

The `frontend/` folder is a separate Python UI built with Streamlit. It is intentionally isolated so a React or Vue app can replace it later without changing the core MLOps code.

Run it with:

```bash
cd frontend
streamlit run app.py
```

The frontend expects a trained model artifact at:

```text
outputs/models/telecom_churn_classifier.joblib
```

## Running The Project

### Main pipeline checks

```bash
python -m pytest
python -m compileall src tests
```

### Train the model

```bash
python -c "from src.pipelines.train_pipeline import run_training_pipeline; run_training_pipeline()"
```

### Run evaluation

```bash
python -c "from src.pipelines.evaluation_pipeline import run_evaluation_pipeline; run_evaluation_pipeline()"
```

### Run batch scoring

```bash
python -c "from src.pipelines.batch_inference_pipeline import run_batch_inference_pipeline; run_batch_inference_pipeline()"
```

### Run drift monitoring

```bash
python -c "from src.pipelines.monitoring_pipeline import run_monitoring_pipeline; run_monitoring_pipeline()"
```

## Generated Outputs

- `outputs/reports/eda_report.md`
- `outputs/reports/model_evaluation.md`
- `outputs/reports/churn_review_queue.md`
- `outputs/reports/drift_report.md`
- `outputs/reports/production_handoff.md`
- `outputs/reports/production_readiness.md`

These are generated artifacts. The model bundle and scored prediction files are intentionally ignored from version control.

## Technology Stack

- Python
- ZenML
- scikit-learn
- MLflow
- Great Expectations
- Evidently
- Streamlit

## Status

- Problem framing: complete
- Architecture: complete
- Core MLOps implementation: complete
- Frontend: complete
- GitHub-ready cleanup: complete

