"""Physics configs: bearing geometry + condition->rpm + sampling rate per dataset."""

from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    cond_to_rpm: dict
    fs: int
    bearing: dict[str, float] | None = None
    name: str = 'custom'


CWRU_PHYSICS = PhysicsConfig(
    bearing={'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0},
    cond_to_rpm={0: 1797, 1: 1772, 2: 1750, 3: 1730},
    fs=12000,
    name='CWRU-6205',
)

JNU_PHYSICS = PhysicsConfig(
    bearing={'n_balls': 8, 'd_ball': 0.2402, 'd_pitch': 1.319, 'contact_angle': 0.0},
    cond_to_rpm={600: 600, 800: 800, 1000: 1000},
    fs=50000,
    name='JNU-N205',
)

SEU_PHYSICS = PhysicsConfig(
    bearing=None,
    cond_to_rpm={0: 1200, 1: 1800},
    fs=20000,
    name='SEU-Gearbox',
)

XJTUSY_PHYSICS = PhysicsConfig(
    bearing={'n_balls': 8, 'd_ball': 7.94, 'd_pitch': 34.55, 'contact_angle': 0.0},
    cond_to_rpm={2100.0: 2100.0, 2250.0: 2250.0, 2400.0: 2400.0},
    fs=25600,
    name='XJTU-SY-LDK-UER204',
)
