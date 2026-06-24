"""Model training utilities for the telecom churn classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_full_feature_pipeline, build_model_feature_schema
from src.models.baselines import ClassificationMetrics


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation/test split for the churn dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def to_dict(self) -> dict[str, int]:
        return {
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
        }


@dataclass
class TrainedChurnModel:
    """Fitted churn model bundle with basic metadata and convenience methods."""

    pipeline: Pipeline
    model_name: str
    target_column: str
    reference_date: str
    split_strategy: str
    random_seed: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_positive_rate: float
    validation_positive_rate: float
    validation_metrics: ClassificationMetrics
    feature_summary: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Return churn probabilities for the positive class."""
        probabilities = self.pipeline.predict_proba(data)
        positive_class_index = _positive_class_index(self.pipeline)
        return pd.Series(probabilities[:, positive_class_index], index=data.index, name="churn_probability")

    def predict(self, data: pd.DataFrame, *, threshold: float = 0.5) -> pd.Series:
        """Return class predictions using a configurable threshold."""
        scores = self.predict_proba(data)
        return (scores >= threshold).astype(int)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metadata without the fitted pipeline object."""
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "reference_date": self.reference_date,
            "split_strategy": self.split_strategy,
            "random_seed": self.random_seed,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_positive_rate": self.train_positive_rate,
            "validation_positive_rate": self.validation_positive_rate,
            "validation_metrics": self.validation_metrics.to_dict(),
            "feature_summary": self.feature_summary,
        }

    def save(self, output_path: str | Path) -> Path:
        """Persist the fitted model bundle with joblib."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, output_path: str | Path) -> "TrainedChurnModel":
        """Load a persisted fitted model bundle."""
        return joblib.load(Path(output_path))


def split_churn_dataset(data: pd.DataFrame, config: dict[str, Any]) -> DatasetSplit:
    """Split the labeled dataset into train, validation, and test partitions."""
    target_column = config["data"]["target_column"]
    split_config = config["evaluation"]["split"]

    test_size = float(split_config["test_size"])
    validation_size = float(split_config["validation_size"])
    random_seed = int(split_config["random_seed"])

    y = pd.to_numeric(data[target_column], errors="coerce")
    stratify = y if _can_stratify(y) else None

    train_validation_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    train_validation_target = pd.to_numeric(train_validation_data[target_column], errors="coerce")
    validation_fraction_of_remaining = validation_size / (1.0 - test_size)
    validation_stratify = train_validation_target if _can_stratify(train_validation_target) else None

    train_data, validation_data = train_test_split(
        train_validation_data,
        test_size=validation_fraction_of_remaining,
        random_state=random_seed,
        stratify=validation_stratify,
    )

    return DatasetSplit(train=train_data, validation=validation_data, test=test_data)


def train_logistic_regression_model(data: pd.DataFrame, config: dict[str, Any]) -> TrainedChurnModel:
    """Train the first churn classifier on the shared feature pipeline."""
    data_config = config["data"]
    target_column = data_config["target_column"]
    split_config = config["evaluation"]["split"]

    split = split_churn_dataset(data, config)
    reference_date = pd.to_datetime(data["signup_date"], errors="coerce").max()
    if pd.isna(reference_date):
        raise ValueError("Cannot train model because signup_date is invalid in the dataset")

    feature_pipeline = build_full_feature_pipeline(config, reference_date=None)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    pipeline = Pipeline(
        steps=[
            *feature_pipeline.steps,
            ("classifier", model),
        ]
    )

    train_features = split.train.drop(columns=[target_column])
    train_target = pd.to_numeric(split.train[target_column], errors="coerce").astype(int)
    validation_features = split.validation.drop(columns=[target_column])
    validation_target = pd.to_numeric(split.validation[target_column], errors="coerce").astype(int)

    pipeline.fit(train_features, train_target)
    validation_predictions = pipeline.predict(validation_features)
    validation_metrics = _compute_classification_metrics(validation_target, validation_predictions)

    feature_schema = build_model_feature_schema(config)
    feature_summary = {
        "numeric_features": list(feature_schema.raw_numeric),
        "categorical_features": list(feature_schema.categorical),
        "derived_features": list(feature_schema.derived_numeric),
        "monitoring_reference": _build_monitoring_reference(split.train, config),
    }

    return TrainedChurnModel(
        pipeline=pipeline,
        model_name="logistic_regression",
        target_column=target_column,
        reference_date=pd.Timestamp(reference_date).isoformat(),
        split_strategy=str(split_config["strategy"]),
        random_seed=int(split_config["random_seed"]),
        train_rows=int(len(split.train)),
        validation_rows=int(len(split.validation)),
        test_rows=int(len(split.test)),
        train_positive_rate=float(train_target.mean()),
        validation_positive_rate=float(validation_target.mean()),
        validation_metrics=validation_metrics,
        feature_summary=feature_summary,
    )


def _compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    support = int((y_true == 1).sum())
    total = int(len(y_true))
    accuracy = accuracy_score(y_true, y_pred) if total else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return ClassificationMetrics(
        true_negatives=true_negative,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_positives=true_positive,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        support=support,
    )


def _can_stratify(target: pd.Series) -> bool:
    counts = target.value_counts(dropna=True)
    return len(counts) > 1 and int(counts.min()) >= 2


def _positive_class_index(pipeline: Pipeline) -> int:
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 in classes:
        return classes.index(1)
    return len(classes) - 1


def _build_monitoring_reference(train_data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    monitoring_config = config.get("monitoring", {})
    numeric_features = list(monitoring_config.get("key_numeric_features", []))
    categorical_features = list(monitoring_config.get("key_categorical_features", []))

    numeric_reference: dict[str, dict[str, float]] = {}
    for feature in numeric_features:
        if feature not in train_data.columns:
            continue
        values = pd.to_numeric(train_data[feature], errors="coerce")
        numeric_reference[feature] = {
            "mean": float(values.mean()) if values.notna().any() else 0.0,
            "missing_rate": float(values.isna().mean()) if len(values) else 0.0,
        }

    categorical_reference: dict[str, dict[str, float]] = {}
    for feature in categorical_features:
        if feature not in train_data.columns:
            continue
        values = train_data[feature].astype("string").fillna("UNKNOWN")
        counts = values.value_counts(dropna=False, normalize=True)
        categorical_reference[feature] = {str(index): float(value) for index, value in counts.items()}

    return {
        "numeric": numeric_reference,
        "categorical": categorical_reference,
    }


__all__ = [
    "DatasetSplit",
    "TrainedChurnModel",
    "split_churn_dataset",
    "train_logistic_regression_model",
]
