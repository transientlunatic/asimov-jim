"""Prior interface between Asimov's generic prior blueprint and Jim's TOML ``[prior]`` table.

Jim's own prior distributions (``jimgw.core.prior``) cover a fixed set of
types -- ``uniform``, ``gaussian``, ``sine``, ``cosine``, ``power_law``,
``rayleigh`` and ``uniform_sphere`` (see ``jimgw.cli._prior.build_prior``) --
each taking a small, type-specific set of fields. Asimov's generic
:class:`asimov.priors.PriorSpecification` only has direct equivalents for
the first five, so :meth:`JimPriorInterface.convert` only translates those;
``rayleigh`` and ``uniform_sphere`` priors must currently be supplied
directly under a per-production ``jim: prior:`` metadata block instead.

This interface deliberately does *not* translate parameter names (e.g.
Bilby's ``chirp_mass``/``mass_ratio`` to Jim's ``M_c``/``q``, or spin
conventions): Jim's parametrization and reference frames are its own, and
guessing at a mapping risks silently producing a different physical prior
than the one written in the blueprint. Write parameter names in the
blueprint's ``priors:`` block using Jim's own names.
"""

from asimov.priors import PriorInterface

_TYPE_MAP = {
    "uniform": "uniform",
    "gaussian": "gaussian",
    "normal": "gaussian",
    "sine": "sine",
    "cosine": "cosine",
    "powerlaw": "power_law",
    "power_law": "power_law",
}


class JimPriorInterface(PriorInterface):
    """
    Convert an Asimov prior blueprint into the dict Jim's Liquid config
    template renders as the ``[prior]`` TOML table.
    """

    def convert(self):
        """
        Convert the blueprint's priors into Jim's per-parameter prior specs.

        Returns
        -------
        dict
            Mapping of parameter name to a dict with a ``type`` key plus
            whatever fields that type needs (``min``/``max``, ``loc``/
            ``scale``, or ``min``/``max``/``alpha`` for ``power_law``).
            ``sine`` and ``cosine`` take no further fields.
        """
        if self.prior_dict is None:
            return {}

        converted = {}
        for name, value in self.prior_dict.to_dict().items():
            if name == "default":
                continue
            spec = self.prior_dict.get_prior(name)
            if spec is None:
                continue
            converted[name] = self._convert_one(name, spec)
        return converted

    def _convert_one(self, name, spec):
        prior_type = (spec.type or "").lower()
        if prior_type not in _TYPE_MAP:
            raise ValueError(
                f"Cannot convert prior for '{name}': Jim interface does not "
                f"know how to translate type '{spec.type}'. Supported types: "
                f"{sorted(set(_TYPE_MAP.values()))}. Set it directly with a "
                f"'jim: prior: {name}: ...' metadata block instead."
            )
        jim_type = _TYPE_MAP[prior_type]

        if jim_type == "uniform":
            return {"type": "uniform", "min": spec.minimum, "max": spec.maximum}
        if jim_type == "gaussian":
            return {"type": "gaussian", "loc": spec.mu, "scale": spec.sigma}
        if jim_type in ("sine", "cosine"):
            return {"type": jim_type}
        if jim_type == "power_law":
            return {
                "type": "power_law",
                "min": spec.minimum,
                "max": spec.maximum,
                "alpha": spec.alpha,
            }
        raise AssertionError(f"unreachable: unhandled jim_type {jim_type!r}")  # pragma: no cover
