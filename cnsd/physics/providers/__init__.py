"""Physics providers: pluggable, domain-specific fault-verification strategies.

The engine speaks only the PhysicsProvider interface; domains register here. A
new machine class (gear, motor, pump, ...) is added by implementing
PhysicsProvider and registering it - no engine change.
"""

from cnsd.physics.providers.base import PhysicsProvider
from cnsd.physics.providers.bearing import BearingProvider
from cnsd.physics.providers.spectral import SpectralProvider

# domain.type (from config) -> provider builder. 'spectral' is the universal
# zero-knowledge fallback used when a domain has no dedicated provider.
_REGISTRY = {
    'bearing': BearingProvider,
    'spectral': SpectralProvider,
    None: SpectralProvider,
}


def register_provider(domain_type, provider_cls):
    """Register a new domain provider (e.g. 'gear' -> GearProvider)."""
    _REGISTRY[domain_type] = provider_cls


def get_provider(domain_type):
    """Return the provider class for a domain type, falling back to spectral."""
    return _REGISTRY.get(domain_type, SpectralProvider)


def available_domains():
    return [k for k in _REGISTRY if k is not None]
