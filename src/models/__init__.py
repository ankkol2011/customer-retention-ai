"""Model training, evaluation, and registry utilities."""

from src.models.baselines import (
    ClassificationMetrics,
    SupportTicketBaselineResult,
    apply_support_ticket_rule,
    evaluate_support_ticket_rule,
)
from src.models.evaluation import EvaluationReport, SliceMetric, evaluate_trained_churn_model
from src.models.inference import BatchPredictionReport, score_batch_predictions
from src.models.training import DatasetSplit, TrainedChurnModel, split_churn_dataset, train_logistic_regression_model

__all__ = [
    "ClassificationMetrics",
    "DatasetSplit",
    "EvaluationReport",
    "BatchPredictionReport",
    "SupportTicketBaselineResult",
    "SliceMetric",
    "TrainedChurnModel",
    "apply_support_ticket_rule",
    "evaluate_support_ticket_rule",
    "evaluate_trained_churn_model",
    "score_batch_predictions",
    "split_churn_dataset",
    "train_logistic_regression_model",
]
