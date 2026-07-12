import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    base_dir: str
    sampling_rate_hz: int
    features: list[str]

    def __post_init__(self):
        if not isinstance(self.sampling_rate_hz, int) or self.sampling_rate_hz <= 0:
            raise ValueError(
                f'sampling_rate_hz must be a positive integer. Got: {self.sampling_rate_hz}'
            )
        if not isinstance(self.features, list) or len(self.features) == 0:
            raise ValueError('features must be a non-empty list of strings.')


@dataclass
class TargetConfig:
    column: str

    def __post_init__(self):
        if not self.column:
            raise ValueError('Target column cannot be empty.')


@dataclass
class DomainConfig:
    type: str


@dataclass
class PhysicsConfigParameters:
    parameters: dict[str, Any]


@dataclass
class TaxonomyConfig:
    classes: dict[int, Any]


@dataclass
class CNSDConfig:
    schema_version: str
    dataset: DatasetConfig
    target: TargetConfig
    domain: DomainConfig
    taxonomy: TaxonomyConfig
    physics: PhysicsConfigParameters | None = None

    def __post_init__(self):
        if self.schema_version != '1.0':
            raise ValueError(
                f"Unsupported schema_version: '{self.schema_version}'. Expected '1.0'."
            )


def load_config(config_path: str | None = None) -> CNSDConfig:
    """
    Loads the generic configuration from a YAML file.
    Can be overridden via the CNSD_CONFIG environment variable.
    """
    if config_path is None:
        # Default to the current working directory's cnsd_config.yaml,
        # or allow override via environment variable for CI/CD and deployment
        config_path = os.getenv('CNSD_CONFIG', 'cnsd_config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found at {config_path}')

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Extract physics safely, allowing it to be None for non-bearing domains
    physics_raw = raw_config.get('physics')
    physics_obj = None
    if physics_raw is not None and 'parameters' in physics_raw:
        physics_obj = PhysicsConfigParameters(parameters=physics_raw['parameters'])

    return CNSDConfig(
        schema_version=str(raw_config.get('schema_version', '1.0')),
        dataset=DatasetConfig(**raw_config.get('dataset', {})),
        target=TargetConfig(**raw_config.get('target', {})),
        domain=DomainConfig(**raw_config.get('domain', {})),
        physics=physics_obj,
        taxonomy=TaxonomyConfig(classes=raw_config.get('taxonomy', {}).get('classes', {})),
    )
