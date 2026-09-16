"""Tests for the Jim prior interface."""

import pytest

from asimov_jim.priors import JimPriorInterface


class TestJimPriorInterface:
    def test_convert_none_returns_empty_dict(self):
        interface = JimPriorInterface(None)
        assert interface.convert() == {}

    def test_convert_uniform(self):
        interface = JimPriorInterface(
            {"M_c": {"type": "uniform", "minimum": 10.0, "maximum": 80.0}}
        )
        assert interface.convert() == {
            "M_c": {"type": "uniform", "min": 10.0, "max": 80.0}
        }

    def test_convert_gaussian(self):
        interface = JimPriorInterface(
            {"psi": {"type": "gaussian", "mu": 0.0, "sigma": 1.0}}
        )
        assert interface.convert() == {
            "psi": {"type": "gaussian", "loc": 0.0, "scale": 1.0}
        }

    def test_convert_sine_and_cosine_take_no_extra_fields(self):
        interface = JimPriorInterface(
            {"iota": {"type": "sine"}, "dec": {"type": "cosine"}}
        )
        assert interface.convert() == {
            "iota": {"type": "sine"},
            "dec": {"type": "cosine"},
        }

    def test_convert_power_law(self):
        interface = JimPriorInterface(
            {"d_L": {"type": "power_law", "minimum": 1.0, "maximum": 2000.0, "alpha": 2.0}}
        )
        assert interface.convert() == {
            "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0}
        }

    def test_convert_powerlaw_alias(self):
        interface = JimPriorInterface(
            {"d_L": {"type": "PowerLaw", "minimum": 1.0, "maximum": 2000.0, "alpha": 2.0}}
        )
        assert interface.convert()["d_L"]["type"] == "power_law"

    def test_convert_unknown_type_raises_clear_error(self):
        interface = JimPriorInterface({"a_1": {"type": "truncated_gaussian"}})
        with pytest.raises(ValueError, match="does not know how to translate"):
            interface.convert()

    def test_convert_skips_default_key(self):
        interface = JimPriorInterface(
            {"default": "BBHPriorDict", "iota": {"type": "sine"}}
        )
        assert interface.convert() == {"iota": {"type": "sine"}}

    def test_convert_multiple_parameters(self):
        interface = JimPriorInterface(
            {
                "M_c": {"type": "uniform", "minimum": 10.0, "maximum": 80.0},
                "iota": {"type": "sine"},
            }
        )
        converted = interface.convert()
        assert set(converted.keys()) == {"M_c", "iota"}
