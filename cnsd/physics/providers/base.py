"""The physics provider interface.

A provider answers three domain-neutral questions for one signal window, letting
the diagnosis engine verify a neural prediction against physics without knowing
what kind of machine produced the signal. Bearings, gears, motors, or a generic
spectral fallback all implement the same contract.
"""

from abc import ABC, abstractmethod
from typing import Any


class PhysicsProvider(ABC):
    """Domain-neutral physics-verification contract.

    Implementations recompute the physical fault signatures for their machine
    class and report which fault family the signal evidence supports. The engine
    consumes only this interface and never references a specific domain.
    """

    #: fault families this provider can evidence, e.g. ['Outer Race', ...].
    families: list[str] = []

    @abstractmethod
    def evidence(self, signal, condition) -> dict[str, Any]:
        """Physical evidence for one window.

        Returns a dict with at least:
          'family_strength' : {family: float}  prominence per fault family
          'frequencies_hz'  : {name: float}    characteristic frequencies (may be {})
        """

    @abstractmethod
    def root_cause(self, family: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Structured root cause for a fault family given the evidence."""

    def dominant_family(self, evidence: dict[str, Any]) -> str | None:
        """The fault family the physical evidence most supports."""
        fs = evidence.get('family_strength', {})
        return max(fs, key=fs.get) if fs else None

    @property
    def name(self) -> str:
        return self.__class__.__name__
