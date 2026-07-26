"""
normal_events.py — Sample normal (benign) events from entity profiles.

For each entity, for each simulated day, we:
  1. Draw event count from Poisson(entity.daily_event_rate)
  2. Sample login hour from Beta(alpha, beta) → mapped to [0, 24)
  3. Pick source_ip from the entity's IP pool
  4. Pick geo_location from home geos
  5. Pick resource_accessed from typical resources (with ~1% curiosity)
  6. Sample session_duration from LogNormal(mu, sigma)
  7. Generate command_sequence from command vocabulary
  8. Pick device_fingerprint from known devices
  9. Set auth_method from weighted distribution
  10. All labeled "normal"
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def _sample_hour(profile: EntityProfile, rng: np.random.Generator) -> float:
    """Sample an hour-of-day from the entity's Beta distribution."""
    raw = rng.beta(profile.hour_alpha, profile.hour_beta)
    hour = profile.hour_shift + raw * profile.hour_scale
    # Wrap around midnight
    return hour % 24.0


def _sample_command_sequence(
    profile: EntityProfile, rng: np.random.Generator
) -> str:
    """Generate a realistic command sequence (3–8 commands)."""
    cmds = list(profile.command_weights.keys())
    weights = np.array(list(profile.command_weights.values()))
    weights = weights / weights.sum()

    seq_len = rng.integers(3, 9)
    chosen = rng.choice(cmds, size=seq_len, p=weights)
    return ",".join(chosen)


def _pick_auth_method(profile: EntityProfile, rng: np.random.Generator) -> str:
    """Pick auth method based on entity's weighted distribution."""
    methods = list(profile.auth_methods.keys())
    weights = np.array(list(profile.auth_methods.values()))
    weights = weights / weights.sum()
    return str(rng.choice(methods, p=weights))


def _format_geo(geo: dict) -> str:
    """Format a geo dict to 'City, CC' string."""
    return f"{geo['city']}, {geo['country']}"


def sample_normal_events(
    profile: EntityProfile,
    config: GeneratorConfig,
    start_date: datetime,
    rng: np.random.Generator,
) -> List[dict]:
    """Sample all normal events for a single entity over the simulation window."""
    events = []

    for day_offset in range(config.sim_days):
        current_date = start_date + timedelta(days=day_offset)

        # Number of events today
        n_events = rng.poisson(profile.daily_event_rate)

        for _ in range(n_events):
            # Timestamp
            hour = _sample_hour(profile, rng)
            hour_int = int(hour)
            minute = int((hour - hour_int) * 60)
            second = rng.integers(0, 60)
            ts = current_date.replace(
                hour=hour_int % 24, minute=minute, second=int(second),
                microsecond=0,
            )
            # Add small random jitter (±30 min) for realism
            jitter = timedelta(seconds=int(rng.normal(0, 300)))
            ts = ts + jitter
            # Clamp to same day
            ts = ts.replace(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
            )

            # Source IP: from entity's pool (98%) or random nearby (2%)
            if rng.random() < 0.98 and profile.source_ips:
                source_ip = str(rng.choice(profile.source_ips))
            else:
                source_ip = f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}"

            # Geo location: from home geos
            geo = profile.home_geos[rng.integers(len(profile.home_geos))]
            geo_str = _format_geo(geo)

            # Resource: from typical set (99%) or curiosity (1%)
            if rng.random() < 0.99 and profile.typical_resources:
                weights = np.array(profile.resource_weights)
                weights = weights / weights.sum()
                resource = str(rng.choice(profile.typical_resources, p=weights))
            else:
                resource = str(rng.choice(config.resource_pool))

            # Session duration
            duration = float(rng.lognormal(profile.session_mu, profile.session_sigma))
            duration = round(max(0.1, duration), 2)  # min 6 seconds

            # Command sequence
            cmd_seq = _sample_command_sequence(profile, rng)

            # Device fingerprint
            device_fp = str(rng.choice(profile.device_fingerprints))

            # Auth method
            auth = _pick_auth_method(profile, rng)

            events.append({
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts,
                "source_ip": source_ip,
                "geo_location": geo_str,
                "resource_accessed": resource,
                "auth_method": auth,
                "session_duration": duration,
                "command_sequence": cmd_seq,
                "device_fingerprint": device_fp,
                "label": "normal",
                "attack_subtype": None,
            })

    return events


def generate_all_normal_events(
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    start_date: datetime,
) -> pd.DataFrame:
    """Generate normal events for all entities and return a DataFrame."""
    rng = np.random.default_rng(config.random_seed + 1000)  # offset seed
    all_events = []

    for profile in profiles:
        events = sample_normal_events(profile, config, start_date, rng)
        all_events.extend(events)

    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
