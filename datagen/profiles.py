"""
profiles.py — Entity behavioral profile generation.

Each entity (user, service_account, edge_device) gets a unique behavioral
profile *before* any events are sampled.  This profile defines:
  - Habitual login hours (Beta distribution parameters)
  - Home geo-locations (1–2 for users, 1 for others)
  - Typical resource set (subset of the global resource pool)
  - Session duration distribution (LogNormal parameters)
  - Auth method weights
  - Known device fingerprints
  - Daily event rate (Poisson λ)
  - Command vocabulary weights
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from datagen.config import GeneratorConfig


@dataclass
class EntityProfile:
    """Behavioral profile for a single entity."""

    entity_id: str
    entity_type: str  # "user" | "service_account" | "edge_device"

    # Hour-of-day distribution: Beta(a, b) mapped to [0, 24)
    hour_alpha: float = 5.0
    hour_beta: float = 2.0
    hour_shift: float = 6.0   # shift so peak is around work hours
    hour_scale: float = 14.0  # scale to span work-hour range

    # Home geolocations: list of {city, country, lat, lon}
    home_geos: List[dict] = field(default_factory=list)

    # Typical resources (subset of config.resource_pool)
    typical_resources: List[str] = field(default_factory=list)
    resource_weights: List[float] = field(default_factory=list)

    # Session duration: LogNormal(mu, sigma) in minutes
    session_mu: float = 3.4    # ln(30) ≈ 3.4 → median ~30 min
    session_sigma: float = 0.5

    # Auth methods: {method: weight}
    auth_methods: Dict[str, float] = field(default_factory=dict)

    # Known device fingerprints
    device_fingerprints: List[str] = field(default_factory=list)

    # Daily event rate (Poisson λ)
    daily_event_rate: float = 10.0

    # Command vocabulary: {cmd: weight}
    command_weights: Dict[str, float] = field(default_factory=dict)

    # Source IP pool (generated from home geos)
    source_ips: List[str] = field(default_factory=list)


def _generate_ip_pool(rng: np.random.Generator, n: int = 5) -> List[str]:
    """Generate a small pool of realistic private/public IPs for an entity."""
    ips = []
    # Mix of private and public-looking IPs
    prefixes = ["10.0.", "172.16.", "192.168.", "203.0.", "198.51."]
    prefix = rng.choice(prefixes)
    for _ in range(n):
        ip = f"{prefix}{rng.integers(1, 255)}.{rng.integers(1, 255)}"
        ips.append(ip)
    return ips


def _generate_fingerprint(rng: np.random.Generator) -> str:
    """Generate a hex device fingerprint."""
    chars = "0123456789abcdef"
    fp = "fp_" + "".join(rng.choice(list(chars)) for _ in range(8))
    return fp


def generate_user_profile(
    entity_id: str, config: GeneratorConfig, rng: np.random.Generator
) -> EntityProfile:
    """Generate a behavioral profile for a human user."""
    # Login hours: Beta centered on work hours with per-user variance
    alpha = rng.uniform(3.0, 8.0)
    beta = rng.uniform(2.0, 6.0)
    shift = rng.uniform(5.0, 8.0)
    scale = rng.uniform(12.0, 16.0)

    # 1–2 home geos
    n_geos = rng.choice([1, 2], p=[0.7, 0.3])
    home_geos = [
        config.geo_locations[i]
        for i in rng.choice(len(config.geo_locations), size=n_geos, replace=False)
    ]

    # 3–8 typical resources
    n_resources = rng.integers(3, 9)
    resource_indices = rng.choice(len(config.resource_pool), size=n_resources, replace=False)
    typical_resources = [config.resource_pool[i] for i in resource_indices]
    # Zipf-like weights: some resources used much more than others
    raw_weights = rng.zipf(1.5, size=n_resources).astype(float)
    resource_weights = (raw_weights / raw_weights.sum()).tolist()

    # Session duration: median 15–60 min
    session_mu = rng.uniform(2.7, 4.1)  # ln(15)=2.7, ln(60)=4.1
    session_sigma = rng.uniform(0.3, 0.7)

    # Auth methods
    auth_methods = dict(config.user_auth_methods)

    # 1–3 devices
    n_devices = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    device_fingerprints = [_generate_fingerprint(rng) for _ in range(n_devices)]

    # Daily rate: 5–20 events
    daily_rate = rng.uniform(5.0, 20.0)

    # Command weights from user command vocab
    cmds = list(config.user_commands)
    raw_w = rng.dirichlet(np.ones(len(cmds)) * 0.5)
    command_weights = {c: float(w) for c, w in zip(cmds, raw_w)}

    return EntityProfile(
        entity_id=entity_id,
        entity_type="user",
        hour_alpha=alpha,
        hour_beta=beta,
        hour_shift=shift,
        hour_scale=scale,
        home_geos=home_geos,
        typical_resources=typical_resources,
        resource_weights=resource_weights,
        session_mu=session_mu,
        session_sigma=session_sigma,
        auth_methods=auth_methods,
        device_fingerprints=device_fingerprints,
        daily_event_rate=daily_rate,
        command_weights=command_weights,
        source_ips=_generate_ip_pool(rng),
    )


def generate_service_account_profile(
    entity_id: str, config: GeneratorConfig, rng: np.random.Generator
) -> EntityProfile:
    """Generate a behavioral profile for an automated service account."""
    # Near-uniform hours (24/7) — flat Beta
    alpha = rng.uniform(1.0, 2.0)
    beta = rng.uniform(1.0, 2.0)
    shift = 0.0
    scale = 24.0

    # Single datacenter geo
    home_geos = [config.geo_locations[rng.integers(len(config.geo_locations))]]

    # 2–5 specific APIs/DBs
    n_resources = rng.integers(2, 6)
    resource_indices = rng.choice(len(config.resource_pool), size=n_resources, replace=False)
    typical_resources = [config.resource_pool[i] for i in resource_indices]
    raw_weights = rng.dirichlet(np.ones(n_resources) * 2.0)
    resource_weights = raw_weights.tolist()

    # Short sessions: median 2–10 min
    session_mu = rng.uniform(0.7, 2.3)  # ln(2)=0.7, ln(10)=2.3
    session_sigma = rng.uniform(0.2, 0.4)

    auth_methods = dict(config.service_auth_methods)

    # Single server fingerprint
    device_fingerprints = [_generate_fingerprint(rng)]

    # High daily rate: 50–200
    daily_rate = rng.uniform(50.0, 200.0)

    cmds = list(config.service_account_commands)
    raw_w = rng.dirichlet(np.ones(len(cmds)) * 0.5)
    command_weights = {c: float(w) for c, w in zip(cmds, raw_w)}

    return EntityProfile(
        entity_id=entity_id,
        entity_type="service_account",
        hour_alpha=alpha,
        hour_beta=beta,
        hour_shift=shift,
        hour_scale=scale,
        home_geos=home_geos,
        typical_resources=typical_resources,
        resource_weights=resource_weights,
        session_mu=session_mu,
        session_sigma=session_sigma,
        auth_methods=auth_methods,
        device_fingerprints=device_fingerprints,
        daily_event_rate=daily_rate,
        command_weights=command_weights,
        source_ips=_generate_ip_pool(rng, n=2),
    )


def generate_edge_device_profile(
    entity_id: str, config: GeneratorConfig, rng: np.random.Generator
) -> EntityProfile:
    """Generate a behavioral profile for an IoT/OT edge device."""
    # Concentrated around shift times (e.g., 6–18) with some 24/7
    alpha = rng.uniform(2.0, 5.0)
    beta = rng.uniform(2.0, 5.0)
    shift = rng.uniform(4.0, 7.0)
    scale = rng.uniform(10.0, 16.0)

    # Single fixed site
    home_geos = [config.geo_locations[rng.integers(len(config.geo_locations))]]

    # 1–3 sensors/endpoints
    # Prefer OT/IoT resources
    ot_resources = [r for r in config.resource_pool if any(
        k in r for k in ["plc", "scada", "sensor", "gateway", "actuator"]
    )]
    n_resources = min(rng.integers(1, 4), len(ot_resources))
    resource_indices = rng.choice(len(ot_resources), size=n_resources, replace=False)
    typical_resources = [ot_resources[i] for i in resource_indices]
    raw_weights = rng.dirichlet(np.ones(n_resources) * 2.0)
    resource_weights = raw_weights.tolist()

    # Very short sessions: median 1–5 min
    session_mu = rng.uniform(0.0, 1.6)  # ln(1)=0, ln(5)=1.6
    session_sigma = rng.uniform(0.2, 0.5)

    auth_methods = dict(config.edge_auth_methods)

    device_fingerprints = [_generate_fingerprint(rng)]

    # Moderate rate: 10–40
    daily_rate = rng.uniform(10.0, 40.0)

    cmds = list(config.edge_device_commands)
    raw_w = rng.dirichlet(np.ones(len(cmds)) * 0.5)
    command_weights = {c: float(w) for c, w in zip(cmds, raw_w)}

    return EntityProfile(
        entity_id=entity_id,
        entity_type="edge_device",
        hour_alpha=alpha,
        hour_beta=beta,
        hour_shift=shift,
        hour_scale=scale,
        home_geos=home_geos,
        typical_resources=typical_resources,
        resource_weights=resource_weights,
        session_mu=session_mu,
        session_sigma=session_sigma,
        auth_methods=auth_methods,
        device_fingerprints=device_fingerprints,
        daily_event_rate=daily_rate,
        command_weights=command_weights,
        source_ips=_generate_ip_pool(rng, n=2),
    )


def generate_all_profiles(config: GeneratorConfig) -> List[EntityProfile]:
    """Generate behavioral profiles for all entities."""
    rng = np.random.default_rng(config.random_seed)
    profiles = []

    for i in range(config.num_users):
        eid = f"user_{i:03d}"
        profiles.append(generate_user_profile(eid, config, rng))

    for i in range(config.num_service_accounts):
        eid = f"svc_{i:03d}"
        profiles.append(generate_service_account_profile(eid, config, rng))

    for i in range(config.num_edge_devices):
        eid = f"edge_{i:03d}"
        profiles.append(generate_edge_device_profile(eid, config, rng))

    return profiles
