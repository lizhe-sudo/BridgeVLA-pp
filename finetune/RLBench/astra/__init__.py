"""Interfaces for Astra RLBench evaluation.

Keep package import lightweight so ``eval_astra.py --help`` does not import
NumPy, RLBench, or compiled simulator dependencies.
"""

__all__ = [
    "AstraAction", "AstraActionAdapter", "AstraObservation",
    "AstraObservationAdapter", "AstraPolicy",
]


def __getattr__(name):
    if name in ("AstraAction", "AstraObservation"):
        from . import schemas
        return getattr(schemas, name)
    if name == "AstraActionAdapter":
        from .action_adapter import AstraActionAdapter
        return AstraActionAdapter
    if name == "AstraObservationAdapter":
        from .observation_adapter import AstraObservationAdapter
        return AstraObservationAdapter
    if name == "AstraPolicy":
        from .policy import AstraPolicy
        return AstraPolicy
    raise AttributeError(name)
