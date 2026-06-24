"""Shared preprocessing pipeline for churn model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.feature_engineering import add_derived_features


@dataclass(frozen=True)
class ModelFeatureSchema:
    """Feature groups used by the churn preprocessing pipeline."""

    raw_numeric: list[str]
    categorical: list[str]
    date: list[str]
    text: list[str]
    derived_numeric: list[str]
    excluded: list[str]

    @property
    def numeric_for_preprocessing(self) -> list[str]:
        return _dedupe_preserve_order(*(self.raw_numeric + self.derived_numeric))

    @property
    def model_columns(self) -> list[str]:
        return _dedupe_preserve_order(*(self.numeric_for_preprocessing + self.categorical))


def build_model_feature_schema(config: dict[str, Any]) -> ModelFeatureSchema:
    """Build the feature schema from project config."""
    feature_config = config["features"]
    derived_config = feature_config["derived"]

    feedback_prefix = derived_config.get("feedback_keyword_prefix", "customer_feedback_keyword")
    derived_numeric = _dedupe_preserve_order(
        derived_config["tenure_column"],
        derived_config["support_ticket_rule_column"],
        derived_config["feedback_length_column"],
        f"{feedback_prefix}_billing",
        f"{feedback_prefix}_coverage",
        f"{feedback_prefix}_support",
    )

    return ModelFeatureSchema(
        raw_numeric=list(feature_config.get("numeric", [])),
        categorical=list(feature_config.get("categorical", [])),
        date=list(feature_config.get("date", [])),
        text=list(feature_config.get("text", [])),
        derived_numeric=derived_numeric,
        excluded=list(feature_config.get("excluded_from_training", [])),
    )


def build_feature_engineering_transformer(
    config: dict[str, Any],
    *,
    reference_date: str | pd.Timestamp | None = None,
) -> FunctionTransformer:
    """Build a transformer that adds the derived churn features."""
    return FunctionTransformer(
        func=partial(add_derived_features, config=config, reference_date=reference_date),
        validate=False,
    )


def build_preprocessing_pipeline(config: dict[str, Any]) -> ColumnTransformer:
    """Build the column transformer applied after feature engineering."""
    schema = build_model_feature_schema(config)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema.numeric_for_preprocessing),
            ("categorical", categorical_pipeline, schema.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_feature_pipeline(
    config: dict[str, Any],
    *,
    reference_date: str | pd.Timestamp | None = None,
) -> Pipeline:
    """Build the end-to-end feature pipeline used for training and scoring."""
    return Pipeline(
        steps=[
            ("feature_engineering", build_feature_engineering_transformer(config, reference_date=reference_date)),
            ("preprocessing", build_preprocessing_pipeline(config)),
        ]
    )


def _dedupe_preserve_order(*items: str) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_one_hot_encoder() -> OneHotEncoder:
    """Build a dense one-hot encoder compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


__all__ = [
    "ModelFeatureSchema",
    "build_feature_engineering_transformer",
    "build_full_feature_pipeline",
    "build_model_feature_schema",
    "build_preprocessing_pipeline",
]
