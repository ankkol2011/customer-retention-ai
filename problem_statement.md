# Problem Statement: Telecom Customer Churn Retention

## Business Context

The business goal is to reduce avoidable telecom customer churn by identifying customers who are likely to leave, then sending those customers to a retention team for review.

The model is not the final decision-maker. It produces a prioritized list of customers who may need attention. A human retention specialist decides whether to contact the customer and what action, if any, is appropriate.

## ML Formulation

- **Problem type**: Binary classification
- **Target variable**: `churn`
- **Positive class**: `1`, meaning the customer churned
- **Negative class**: `0`, meaning the customer did not churn
- **Prediction output**: Churn risk score and review priority for each customer

The system should rank customers by churn risk instead of only producing a hard yes/no label. A ranked list is better for retention operations because humans usually have limited time and need to review the highest-risk customers first.

## Primary Metric

- **Primary metric**: Recall for the churn class

False negatives are the more costly mistake: if the model predicts that a customer will stay but the customer actually churns, the retention team misses the chance to intervene.

Recall answers the key business question: of all customers who actually churn, how many did we catch?

## Guardrail Metrics

- **Precision**: Keeps the human review queue useful by measuring how many flagged customers are truly at risk.
- **PR-AUC**: Measures ranking quality for the churn class and is more useful than accuracy when the positive class matters most.
- **Calibration**: Checks whether predicted risk scores behave like real probabilities, which matters when humans use those scores to make decisions.

Accuracy is not the primary metric because it can hide missed churners. A model can look accurate while still failing the business goal.

## Data Summary

- **Dataset path**: `data/telecom_customer_churn_feature_engineering.csv`
- **Rows**: 1,200
- **Columns**: 15
- **Missing values**: None detected during initial read-only inspection
- **Churn distribution**:
  - `0`: 780 customers, 65%
  - `1`: 420 customers, 35%

### Feature Groups

- **Identifier**: `customer_id`
- **Date**: `signup_date`
- **Numeric features**: `age`, `monthly_income`, `monthly_bill`, `internet_usage_gb`, `call_minutes`, `support_tickets`
- **Categorical features**: `gender`, `city`, `education_level`, `employment_status`, `contract_type`
- **Text feature**: `customer_feedback`
- **Target**: `churn`

`customer_id` should not be used as a predictive feature because it is an identifier, not customer behavior. `signup_date` should be transformed into tenure-style features before modeling. `customer_feedback` may be useful, but it should be handled carefully so the first production baseline stays explainable and reliable.

## Current Baseline

The current decision process is rule-based: customers are flagged for outreach using triggers such as many support tickets.

The ML system must be compared against this baseline. A useful model should catch churners that the rule-based trigger misses while keeping the review queue manageable for humans.

## Constraints

- **Human-in-the-loop**: Required. The model recommends and ranks; humans decide.
- **Prediction mode**: Batch inference for the first production version.
- **Interpretability**: Required for retention review. Humans need simple reasons for why a customer was flagged.
- **Production priority**: Reproducible pipelines, validation, model versioning, monitoring, and rollback paths.
- **Dataset limitation**: 1,200 rows is enough for an MVP baseline but small for a mature production churn model. The system should be designed to accept fresher and larger operational data later.

## Framework

- **Orchestration framework**: ZenML

ZenML will be used to structure the training and batch inference workflows into repeatable, versioned pipelines.

## Success Criteria

The first production-ready version is successful when:

1. A reproducible ZenML training pipeline can load data, validate it, transform features, train a baseline model, evaluate it, and register the model artifact.
2. The model improves on the rule-based support-ticket baseline for finding churners.
3. Recall for churners is optimized while precision remains high enough that the retention team can realistically review the output.
4. A batch inference pipeline produces a ranked human-review list with customer IDs, churn risk scores, priority bands, and simple explanations.
5. Model runs, metrics, data assumptions, and artifacts are versioned so the system can be audited and rolled back.

## MVP Scope

The MVP should focus on a reliable, explainable baseline before adding complex automation.

Initial scope:

- Data loading and validation
- Feature preprocessing using reusable training/inference logic
- Baseline model training
- Evaluation against recall-focused metrics
- Batch prediction output for human review
- Model and metric tracking through the MLOps stack

Deferred until after the baseline is working:

- Automated retraining
- Real-time API serving
- Advanced text modeling for `customer_feedback`
- Full production alerting and incident response

