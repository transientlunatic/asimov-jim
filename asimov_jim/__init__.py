"""Asimov integration for the Jim (JimGW) parameter estimation pipeline."""

from .pipeline import Jim

__all__ = ["Jim"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("asimov-jim")
except PackageNotFoundError:  # pragma: no cover - only hit when not installed
    __version__ = "0.0.0"
