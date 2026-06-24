# Telecom Customer Churn MLOps

Production-minded batch ML system for telecom churn risk ranking.

The model is advisory: it produces a retention review list for humans. It does not automatically contact customers or assign offers.

## Current Phase

- Phase 1 problem framing: complete
- Phase 2 architecture: complete and approved
- Phase 3 implementation: hardening and handoff in progress

## Core Artifacts

- `problem_statement.md`: business and ML framing
- `architecture.md`: approved MLOps architecture
- `configs/project.yaml`: project paths, features, metrics, and validation settings

## MVP Stack

- ZenML local orchestrator
- Local artifact store
- MLflow experiment tracking and model registry
- Great Expectations plus custom validation checks
- Evidently drift reports after the baseline pipeline works

## Dataset

Initial training data:

```text
data/telecom_customer_churn_feature_engineering.csv
```

Target:

```text
churn
```

Positive class:

```text
1
```

## Planned Pipelines

- Training pipeline: load data, validate, build features, train, evaluate, register.
- Batch inference pipeline: load approved model, score CSV report, write ranked retention list.
- Monitoring pipeline: summarize batch health, prediction health, and delayed label performance.
- Drift pipeline: compare new reports and prediction distributions against training baselines.

## Implemented Pipelines

- Training pipeline: load validated CSV, fit the shared feature pipeline, train logistic regression, save the bundle.
- Evaluation pipeline: compare the trained model against the support-ticket baseline and write a Markdown report.
- Batch inference pipeline: score a CSV report, rank customers by churn risk, and persist the review queue.
- Monitoring pipeline: compare current batch features and predictions against the training reference.

## Hardening

- `src/verification.py` checks that saved model and batch artifacts are loadable and internally consistent.
- `outputs/reports/production_handoff.md` summarizes the production contract for the current state of the project.
- `.github/workflows/ci.yml` runs syntax checks and the full test suite on push and pull request.
- `outputs/reports/production_readiness.md` records the current readiness checklist and residual risks.

## How to Run

```bash
python -m pytest
python -m compileall src tests
```

The trained model, evaluation report, batch review queue, and drift report are all produced through the pipeline modules under `src/pipelines/`.
