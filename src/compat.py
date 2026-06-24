"""Compatibility helpers for optional runtime dependencies."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def zenml_step(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Return ZenML's step decorator when available, otherwise a no-op decorator."""
    try:
        from zenml import step as zenml_step_impl
    except ModuleNotFoundError:
        def decorator(function: T) -> T:
            return function

        return decorator

    return zenml_step_impl(*args, **kwargs)


def zenml_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Return ZenML's pipeline decorator when available, otherwise a no-op decorator."""
    try:
        from zenml import pipeline as zenml_pipeline_impl
    except ModuleNotFoundError:
        def decorator(function: T) -> T:
            return function

        return decorator

    return zenml_pipeline_impl(*args, **kwargs)
