# Architecture: Telecom Customer Churn MLOps System

## MLOps Pipeline Overview

This system predicts telecom customer churn risk so a human retention team can review high-risk customers and decide outreach actions.

The model is advisory. It ranks customers and explains risk signals; it does not automatically contact customers, assign discounts, or make final retention decisions.

The production lifecycle has ten stages:

1. **Data ingestion**: Load customer CSV reports.
2. **Data validation**: Check schema, types, ranges, missingness, duplicates, and batch volume.
3. **Feature engineering**: Convert raw columns into reusable model features.
4. **Model training**: Train baseline and candidate churn models.
5. **Model evaluation**: Compare models against recall-focused metrics and the support-ticket rule.
6. **Model registry**: Store approved model versions with metadata and lifecycle stage.
7. **Deployment**: Run batch scoring to produce retention review lists.
8. **Monitoring**: Track batch health, feature health, prediction health, and human review outcomes.
9. **Drift detection**: Compare new CSV batches and prediction distributions against training baselines.
10. **Retraining trigger**: Retrain only after human-approved triggers and promotion gates.

The MVP targets MLOps Level 1: reproducible pipelines, validation, versioned artifacts, experiment tracking, and a defined batch inference path. Level 2 promotion gates and deeper monitoring will be added as the system matures.

## Data Plan

Initial data arrives as CSV reports.

- **Current dataset**: `data/telecom_customer_churn_feature_engineering.csv`
- **Rows**: 1,200
- **Columns**: 15
- **Target**: `churn`
- **Positive class**: `1`, meaning customer churned
- **Observed churn rate**: 35%
- **Missing values**: none detected during initial inspection

Recommended storage layout:

```text
data/raw/              # immutable source CSV reports
data/processed/        # optional validated/transformed outputs
outputs/predictions/   # batch retention review lists
```

Each training or scoring run must record the input CSV path and file hash.

Validation checks:

- required columns exist
- expected data types are present
- `churn` exists for training data
- `churn` is absent or ignored for scoring data
- numeric ranges are reasonable
- categorical values are expected or safely handled as unknown
- row count is within expected bounds
- duplicate `customer_id` values are detected
- report freshness is captured

Raw CSVs should be treated as immutable snapshots. If reports become frequent, large, or shared across a team, add DVC or cloud artifact storage.

## Feature Engineering Plan

The same preprocessing logic must be used in training and batch inference.

Excluded from model inputs:

- `customer_id`: identifier only; keep for joins and output files
- `churn`: target only

Numeric features:

- `age`
- `monthly_income`
- `monthly_bill`
- `internet_usage_gb`
- `call_minutes`
- `support_tickets`

Categorical features:

- `gender`
- `city`
- `education_level`
- `employment_status`
- `contract_type`

Date-derived features:

- derive `customer_tenure_days` from `signup_date`
- optionally add `signup_month` or `signup_year`

Text-derived MVP features from `customer_feedback`:

- feedback length
- simple keyword flags or counts for terms such as billing, coverage, support, poor, and issue

Baseline-rule feature:

- `many_support_tickets_flag`
- initial threshold: `support_tickets >= 3`

Preprocessing strategy:

- numeric imputation using train-split medians if future missing values appear
- numeric scaling for logistic regression
- one-hot encoding for low-cardinality categoricals
- unknown-safe handling for unseen future categories
- one reusable fitted preprocessing artifact stored with the model

No feature store is included in the MVP. A feature store becomes useful later if multiple models reuse features, if data sources multiply, or if real-time serving is added.

## Training And Evaluation Plan

Primary business error:

- False negatives are worse than false positives.
- Missing a likely churner means the retention team loses a chance to intervene.

Primary metric:

- recall for `churn = 1`

Guardrail metrics:

- precision
- PR-AUC
- calibration / Brier score
- review queue size

Baselines:

1. **Existing rule baseline**: `support_tickets >= 3`
2. **Simple ML baseline**: logistic regression
3. **Challenger model**: gradient-boosted tree model after the logistic baseline is established

Evaluation split:

- prefer a time-aware split if `signup_date` is a defensible proxy for observation time
- otherwise use stratified split to preserve the 35% churn rate

Threshold strategy:

- do not use default `0.5`
- use top-K ranked review lists based on retention team capacity

Slice evaluation:

- city
- contract type
- employment status
- age group
- support ticket count band

Experiment tracking should log:

- dataset path/hash
- feature list
- split strategy
- preprocessing artifact
- model type and hyperparameters
- recall, precision, PR-AUC, F1, calibration
- slice metrics
- baseline comparison
- trained model artifact

## Deployment Plan

The MVP uses batch inference.

Batch scoring flow:

1. Load latest approved production model.
2. Load latest customer CSV report.
3. Validate CSV schema and data quality.
4. Apply the fitted preprocessing artifact.
5. Generate churn probabilities.
6. Rank customers by churn risk.
7. Write a retention review list.

Prediction output should include:

- `customer_id`
- `churn_risk_score`
- `risk_band`
- `recommended_review_priority`
- `top_reasons`
- `model_version`
- `scored_at`

Deployment strategy:

- start with shadow-style batch deployment
- keep the support-ticket rule as the operational baseline
- run ML scoring in parallel
- compare ML recommendations against the rule
- allow humans to review ML outputs before operational use

Rollback options:

1. regenerate predictions with the previous approved model
2. fall back to the support-ticket rule
3. pause ML scoring until the issue is fixed

## Monitoring And Drift Plan

MVP monitoring focuses on batch, data, feature, prediction, and human-review health.

CSV/data health:

- required columns
- row count
- duplicate `customer_id` count
- missing value rates
- numeric range checks
- unknown category rates
- report freshness

Feature health:

- `monthly_bill`
- `internet_usage_gb`
- `call_minutes`
- `support_tickets`
- `customer_tenure_days`
- `city`
- `contract_type`
- `employment_status`
- text keyword signal rates

Prediction health:

- number of customers scored
- average churn risk
- high/medium/low risk-band counts
- top-K queue size
- prediction score distribution
- sudden score collapse

Human review workflow metrics:

- number of recommended customers reviewed
- number contacted
- number rejected or overridden by humans
- override reasons
- eventual retention/churn outcomes

Drift detection:

- use the training data as the first reference baseline
- use PSI or KS-style checks for numeric features
- use categorical distribution checks for categorical features
- compare prediction distributions against the previous scoring batch and training baseline

Retraining should be human-approved initially. Triggers include recall degradation, precision degradation, significant drift across multiple important features, business changes, or breaking schema changes.

## Versioning And Governance

Every training run must record:

- dataset path and file hash
- schema version
- feature list and feature logic version
- preprocessing artifact version
- model type and hyperparameters
- split method
- random seed
- code version or git commit
- dependency versions
- global metrics
- slice metrics
- support-ticket baseline comparison

Every batch prediction file must record:

- input CSV path/hash
- model version
- preprocessing version
- scoring timestamp
- prediction run ID
- top-K or threshold rule used
- output row count

Model lifecycle stages:

- `experimental`
- `registered`
- `staging`
- `production`
- `archived`

Promotion gates:

- recall beats support-ticket rule
- precision is acceptable for review capacity
- PR-AUC improves over baseline
- critical slice metrics are acceptable
- golden input parity test passes
- prediction distribution is sane
- rollback target exists
- human reviewer approves production promotion

Each production model should have a model card documenting intended use, limitations, training data, metrics, feature groups, slices, owner, and approval date.

No model reaches production unless it is reproducible from data, code, config, preprocessing artifact, and model artifact.

## ZenML Stack Specification

MVP stack:

| Component | Choice | Reason |
|---|---|---|
| ZenML deployment | Local first | Fastest route to reproducible MVP pipelines |
| Orchestrator | Local | CSV batch jobs are small enough for one machine |
| Artifact store | Local | Stores pipeline artifacts, reports, preprocessors, and models |
| Experiment tracker | MLflow | Tracks parameters, metrics, artifacts, and run comparisons |
| Model registry | MLflow | Supports model versions and promotion stages |
| Data validator | Great Expectations + custom checks | Schema/range checks plus churn-specific validation |
| Drift validator | Evidently after baseline works | Drift and prediction distribution reports |
| Model deployer | None | No real-time API in MVP |
| Pipeline deployer | None | No HTTP pipeline endpoint in MVP |
| Container registry | None | Local orchestrator does not require containers |
| Image builder | None | No remote container execution |
| Feature store | None | Not needed for one CSV-batch model |
| Step operator | None | No GPU or distributed compute needed |
| Alerter | Deferred | Add Slack/email after scheduled production batches |
| Log store | Artifact/default logs | Enough for MVP |
| Service connector | None | Needed only when cloud resources are introduced |

Production migration path:

- remote ZenML OSS server or ZenML Pro
- S3/GCS/Azure Blob artifact store
- Kubernetes, SageMaker, Vertex AI, or AzureML orchestrator
- ECR/GCR/ACR container registry
- Slack/email alerter
- Datadog or OpenTelemetry log store
- cloud service connectors

## Pipeline Decomposition

### Training Pipeline

1. Load training CSV.
2. Validate data.
3. Split train/validation/test.
4. Build features and fit preprocessing.
5. Train support-ticket baseline.
6. Train logistic regression baseline.
7. Optionally train tree-based challenger.
8. Evaluate metrics and slices.
9. Compare against baseline.
10. Register candidate model and preprocessing artifact.

### Batch Inference Pipeline

1. Load production model and preprocessing artifact.
2. Load scoring CSV.
3. Validate scoring data.
4. Apply preprocessing.
5. Generate churn risk scores.
6. Rank customers.
7. Add risk bands and top reasons.
8. Write retention review output.
9. Write prediction summary metadata.

### Drift Detection Pipeline

1. Load training reference profile.
2. Load latest scoring batch.
3. Compare feature distributions.
4. Compare prediction distributions.
5. Generate drift report.
6. Flag review or retraining triggers.

### Monitoring Pipeline

1. Collect batch metadata.
2. Compute data health metrics.
3. Compute prediction health metrics.
4. Join delayed churn labels when available.
5. Compute model performance and slice metrics.
6. Produce monitoring summary.

## Project Structure

Recommended MVP structure:

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── predictions/
│   └── reports/
├── src/
│   ├── pipelines/
│   ├── steps/
│   ├── features/
│   ├── validation/
│   ├── models/
│   └── monitoring/
├── configs/
├── tests/
├── problem_statement.md
└── architecture.md
```

## MVP Scope

Build first:

- project structure and dependency setup
- CSV data loading
- schema and quality validation
- feature preprocessing shared between training and inference
- support-ticket rule baseline
- logistic regression baseline
- evaluation reports and slice metrics
- MLflow experiment tracking
- MLflow model registration
- batch inference output
- basic drift and prediction summary reports

Deferred:

- real-time API
- cloud orchestration
- feature store
- automated retraining
- advanced text embeddings
- full alerting and incident response automation
- CI/CD for ML promotion gates

## Implementation Stop Point

Architecture is now defined. Implementation should not begin until this document is reviewed and approved.

