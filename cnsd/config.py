import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class DatasetConfig:
    name: str
    base_dir: str
    sampling_rate_hz: int
    features: List[str]

@dataclass
class TargetConfig:
    column: str

@dataclass
class DomainConfig:
    type: str

@dataclass
class PhysicsConfigParameters:
    parameters: Dict[str, Any]

@dataclass
class TaxonomyConfig:
    classes: Dict[int, Any]

@dataclass
class CNSDConfig:
    dataset: DatasetConfig
    target: TargetConfig
    domain: DomainConfig
    physics: PhysicsConfigParameters
    taxonomy: TaxonomyConfig

def load_config(config_path: Optional[str] = None) -> CNSDConfig:
    """
    Loads the generic configuration from a YAML file. 
    Can be overridden via the CNSD_CONFIG environment variable.
    """
    if config_path is None:
        # Default to the current working directory's cnsd_config.yaml, 
        # or allow override via environment variable for CI/CD and deployment
        config_path = os.getenv("CNSD_CONFIG", "cnsd_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    return CNSDConfig(
        dataset=DatasetConfig(**raw_config.get("dataset", {})),
        target=TargetConfig(**raw_config.get("target", {})),
        domain=DomainConfig(**raw_config.get("domain", {})),
        physics=PhysicsConfigParameters(parameters=raw_config.get("physics", {}).get("parameters", {})),
        taxonomy=TaxonomyConfig(classes=raw_config.get("taxonomy", {}).get("classes", {}))
    )

# Global singleton configuration that downstream layers will import
try:
    config = load_config()
except FileNotFoundError:
    # Graceful fallback during tests or initial setup before config is explicitly provided
    config = None
