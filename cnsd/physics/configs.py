"""Physics configs: bearing geometry + condition->rpm + sampling rate per dataset."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PhysicsConfig:
    cond_to_rpm: Dict
    fs: int
    bearing: Optional[Dict[str, float]] = None
    name: str = 'custom'


CWRU_PHYSICS = PhysicsConfig(
    bearing={'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0},
    cond_to_rpm={0: 1797, 1: 1772, 2: 1750, 3: 1730}, fs=12000, name='CWRU-6205')

JNU_PHYSICS = PhysicsConfig(
    bearing={'n_balls': 8, 'd_ball': 0.2402, 'd_pitch': 1.319, 'contact_angle': 0.0},
    cond_to_rpm={600: 600, 800: 800, 1000: 1000}, fs=50000, name='JNU-N205')

SEU_PHYSICS = None
