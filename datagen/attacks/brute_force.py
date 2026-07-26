"""
brute_force.py — Inject brute-force attack patterns.

Pattern: 10–50 rapid failed authentication attempts from a single source IP
against a single entity within a short window (<5 minutes), optionally
followed by a single successful auth.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def inject_brute_force(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject brute-force attack events into the event stream."""
    rate = config.attack_rates.get("brute_force", 0.005)
    n_target_events = max(1, int(len(events_df) * rate))
    avg_events_per_campaign = 30  # ~10-50 attempts + optional success
    n_attacks = max(1, n_target_events // avg_events_per_campaign)

    attack_events = []

    # Pick random target entities (prefer users — brute force targets humans)
    user_profiles = [p for p in profiles if p.entity_type == "user"]
    if not user_profiles:
        return events_df

    for _ in range(n_attacks):
        target = rng.choice(user_profiles)

        # Pick a random timestamp from the entity's existing events
        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        base_ts = entity_events.sample(1, random_state=int(rng.integers(1e6)))["timestamp"].iloc[0]

        # Attacker IP (outside entity's normal pool)
        attacker_ip = f"45.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"

        # Generate 10–50 failed attempts in rapid succession
        n_attempts = rng.integers(10, 51)
        for attempt_i in range(n_attempts):
            ts = base_ts + timedelta(seconds=int(attempt_i * rng.uniform(2, 8)))

            event = {
                "entity_id": target.entity_id,
                "entity_type": target.entity_type,
                "timestamp": ts,
                "source_ip": attacker_ip,
                "geo_location": _random_geo(config, rng),
                "resource_accessed": "api_auth",
                "auth_method": "password",
                "session_duration": round(float(rng.uniform(0.01, 0.1)), 2),
                "command_sequence": "login_attempt,fail",
                "device_fingerprint": f"fp_brute_{rng.integers(1000,9999)}",
                "label": "anomaly",
                "attack_subtype": "brute_force",
            }
            attack_events.append(event)

        # 30% chance of eventual success after the burst
        if rng.random() < 0.3:
            success_ts = base_ts + timedelta(seconds=int(n_attempts * 6 + rng.integers(5, 30)))
            success_event = {
                "entity_id": target.entity_id,
                "entity_type": target.entity_type,
                "timestamp": success_ts,
                "source_ip": attacker_ip,
                "geo_location": _random_geo(config, rng),
                "resource_accessed": str(rng.choice(target.typical_resources))
                    if target.typical_resources else "api_auth",
                "auth_method": "password",
                "session_duration": round(float(rng.lognormal(2.0, 0.5)), 2),
                "command_sequence": "login_attempt,success,ls,cat,exit",
                "device_fingerprint": f"fp_brute_{rng.integers(1000,9999)}",
                "label": "anomaly",
                "attack_subtype": "brute_force",
            }
            attack_events.append(success_event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df


def _random_geo(config: GeneratorConfig, rng: np.random.Generator) -> str:
    """Pick a random geo location string."""
    geo = config.geo_locations[rng.integers(len(config.geo_locations))]
    return f"{geo['city']}, {geo['country']}"
