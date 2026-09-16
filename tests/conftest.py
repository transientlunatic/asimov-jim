"""Pytest configuration and fixtures."""

import tempfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_production():
    """Create a mock production object for testing."""
    production = MagicMock()
    production.name = "TestProduction"
    production.pipeline = "jim"
    production.category = "analyses"
    production.rundir = "/tmp/test_rundir"
    production.status = "wait"
    production.job_id = None
    production.priors = None

    production.event = MagicMock()
    production.event.name = "GW150914"
    production.event.repository = MagicMock()
    production.event.repository.directory = "/tmp/test_repo"
    production.event.repository.find_prods.return_value = ["TestProduction.toml"]

    production.meta = {
        "event time": 1126259462.4,
        "interferometers": ["H1", "L1"],
        "waveform": {
            "approximant": "IMRPhenomXPHM",
            "reference frequency": 20,
        },
        "likelihood": {
            "minimum frequency": {"H1": 20, "L1": 20},
            "maximum frequency": {"H1": 896, "L1": 896},
        },
        "scheduler": {
            "accounting group": "ligo.dev.o4.cbc.pe.jim",
            "cpus": 2,
        },
    }

    return production


@pytest.fixture
def mock_config(monkeypatch):
    """Mock the asimov config object."""
    config_values = {
        ("general", "rundir_default"): "/tmp/run",
        ("pipelines", "environment"): "/opt/conda",
        ("condor", "user"): "test.user",
    }

    def mock_get(section, key):
        return config_values.get((section, key), "")

    mock_config_module = MagicMock()
    mock_config_module.get = mock_get

    monkeypatch.setattr("asimov_jim.pipeline.config", mock_config_module)
    return mock_config_module


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
