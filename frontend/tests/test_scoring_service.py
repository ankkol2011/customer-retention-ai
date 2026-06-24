"""Tests for the scoring service boundary."""

from __future__ import annotations

import pytest

from frontend.src.churn_ui.services.artifact_service import load_model_bundle


def test_load_model_bundle_errors_when_missing() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_bundle("does-not-exist.joblib")
