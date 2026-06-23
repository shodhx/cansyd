"""Config -> resolved objects bridge.

The single point where a CNSDConfig (from YAML) is turned into the objects the
engine consumes: a PhysicsProvider and a taxonomy. The engine never reads the
config; it only ever sees these resolved objects. Adding a domain means
registering a provider and a geometry resolver here - no engine change.
"""

from cnsd.physics.providers import SpectralProvider, get_provider

# Named bearing geometries, so a config can say bearing_type: "6205-2RS" instead
# of spelling out the geometry. Extend freely; unknown names fall back to
# requiring explicit geometry in the config.
_BEARING_GEOMETRY = {
    '6205-2RS': {'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0},
    '6203': {'n_balls': 8, 'd_ball': 0.2520, 'd_pitch': 1.122, 'contact_angle': 0.0},
    'N205': {'n_balls': 8, 'd_ball': 0.2402, 'd_pitch': 1.319, 'contact_angle': 0.0},
}


# Map common family aliases to the canonical names the providers use.
_FAMILY_ALIASES = {
    'inner': 'Inner Race',
    'inner race': 'Inner Race',
    'ir': 'Inner Race',
    'inner_race': 'Inner Race',
    'outer': 'Outer Race',
    'outer race': 'Outer Race',
    'or': 'Outer Race',
    'outer_race': 'Outer Race',
    'ball': 'Ball',
    'rolling element': 'Ball',
    're': 'Ball',
    'normal': 'Normal',
    'healthy': 'Normal',
    'none': 'Normal',
}


def _canonical_family(name):
    if name is None:
        return None
    return _FAMILY_ALIASES.get(str(name).strip().lower(), name)


def build_taxonomy(config):
    """Resolve config.taxonomy.classes -> {int_label: (family, severity)}.

    Family names are normalized to the canonical names providers use, so a
    config may write 'Inner', 'IR', or 'inner_race' for 'Inner Race'.
    """
    raw = getattr(getattr(config, 'taxonomy', None), 'classes', {}) or {}
    taxonomy = {}
    for label, spec in raw.items():
        if isinstance(spec, (list, tuple)) and spec:
            family = _canonical_family(spec[0])
            severity = _severity_label(spec[1]) if len(spec) > 1 else 'None'
        else:
            family, severity = _canonical_family(spec), 'None'
        taxonomy[int(label)] = (family, severity)
    return taxonomy


def _severity_label(size):
    if size is None:
        return 'None'
    try:
        size = float(size)
    except (TypeError, ValueError):
        return str(size)
    if size <= 0.007:
        return 'Low'
    if size <= 0.014:
        return 'Medium'
    return 'High'


def build_provider(config):
    """Resolve config.domain.type + physics.parameters -> a PhysicsProvider."""
    domain_type = getattr(getattr(config, 'domain', None), 'type', None)
    fs = getattr(getattr(config, 'dataset', None), 'sampling_rate_hz', 12000)
    provider_cls = get_provider(domain_type)

    params = {}
    if getattr(config, 'physics', None) is not None:
        params = getattr(config.physics, 'parameters', {}) or {}

    if provider_cls is SpectralProvider:
        return SpectralProvider(fs=fs)

    if domain_type == 'bearing':
        geometry = _resolve_bearing_geometry(params)
        cond_to_rpm = _coerce_int_keys(params.get('motor_load_rpm', {0: 1797}))
        return provider_cls(bearing=geometry, cond_to_rpm=cond_to_rpm, fs=fs)

    # other registered domains: pass parameters through + fs
    try:
        return provider_cls(fs=fs, **params)
    except TypeError:
        return provider_cls(fs=fs)


def _resolve_bearing_geometry(params):
    if 'geometry' in params:
        return params['geometry']
    bt = params.get('bearing_type')
    if bt in _BEARING_GEOMETRY:
        return _BEARING_GEOMETRY[bt]
    raise ValueError(
        f'Bearing geometry unknown for bearing_type={bt!r}. Add it to '
        f'_BEARING_GEOMETRY or provide physics.parameters.geometry explicitly.'
    )


def _coerce_int_keys(d):
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out
