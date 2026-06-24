"""ZenML pipeline definitions."""

from src.pipelines.batch_inference_pipeline import churn_batch_inference_pipeline, run_batch_inference_pipeline
from src.pipelines.evaluation_pipeline import churn_evaluation_pipeline, run_evaluation_pipeline
from src.pipelines.monitoring_pipeline import churn_monitoring_pipeline, run_monitoring_pipeline
from src.pipelines.train_pipeline import churn_training_pipeline, run_training_pipeline

__all__ = [
    "churn_batch_inference_pipeline",
    "churn_evaluation_pipeline",
    "churn_training_pipeline",
    "churn_monitoring_pipeline",
    "run_batch_inference_pipeline",
    "run_evaluation_pipeline",
    "run_monitoring_pipeline",
    "run_training_pipeline",
]
