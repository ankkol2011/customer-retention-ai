"""Shared request and response schemas for the frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class UploadValidationResult:
    """Validation outcome for uploaded CSV files."""

    passed: bool
    messages: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages": self.messages,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
        }


@dataclass(frozen=True)
class ScoringResult:
    """Output returned after scoring an uploaded CSV."""

    scored_frame: pd.DataFrame
    report_markdown: str
    output_path: Path | None = None

