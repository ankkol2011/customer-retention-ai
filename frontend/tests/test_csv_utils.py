"""Tests for CSV upload helpers."""

from __future__ import annotations

import pandas as pd

from frontend.src.churn_ui.utils.csv_utils import read_uploaded_csv, validate_uploaded_csv


class _UploadedFile:
    def __init__(self, data: str) -> None:
        self._data = data.encode("utf-8")

    def getvalue(self) -> bytes:
        return self._data


def test_read_uploaded_csv_parses_dataframe() -> None:
    uploaded = _UploadedFile("customer_id,age\n1,22\n2,33\n")

    frame = read_uploaded_csv(uploaded)

    assert list(frame.columns) == ["customer_id", "age"]
    assert len(frame) == 2


def test_validate_uploaded_csv_detects_missing_columns() -> None:
    frame = pd.DataFrame({"customer_id": [1], "age": [22]})

    result = validate_uploaded_csv(frame)

    assert not result.passed
    assert "Missing required columns" in result.messages[0]
