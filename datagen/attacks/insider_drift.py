"""
insider_drift.py — Inject insider-drift (ambiguous edge case) patterns.

Pattern: A legitimate entity gradually expands their resource access set
and shifts toward off-hours activity over several weeks.  This is
intentionally ambiguous — could be a curious employee, role change, or
genuine insider threat.  The model should flag it but with lower confidence.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def inject_insider_drift(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject insider-drift attack events."""
    rate = config.attack_rates.get("insider_drift", 0.003)
    n_campaigns = max(1, int(len(events_df) * rate / 15))  # each campaign = ~15 events

    attack_events = []

    user_profiles = [p for p in profiles if p.entity_type == "user"]
    if not user_profiles:
        return events_df

    for _ in range(n_campaigns):
        target = rng.choice(user_profiles)

        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        # Drift spans 2–4 weeks
        drift_weeks = rng.integers(2, 5)
        drift_days = drift_weeks * 7
        drift_start = entity_events["timestamp"].min() + timedelta(
            days=int(rng.integers(7, max(8, config.sim_days - drift_days)))
        )

        # Gradually expand resource set
        unusual_resources = [
            r for r in config.resource_pool
            if r not in target.typical_resources
        ]
        if len(unusual_resources) < 3:
            continue

        # Each week, the insider accesses 1–2 more unusual resources
        current_new_resources: List[str] = []

        for week_i in range(drift_weeks):
            # Add 1–2 new resources this week
            n_new = rng.integers(1, 3)
            for _ in range(n_new):
                if unusual_resources:
                    new_r = str(unusual_resources.pop(rng.integers(len(unusual_resources))))
                    current_new_resources.append(new_r)

            # Generate 3–5 events this week accessing the expanding set
            week_start = drift_start + timedelta(weeks=int(week_i))
            n_events = rng.integers(3, 6)

            for _ in range(n_events):
                day_offset = rng.integers(0, 7)
                ts_day = week_start + timedelta(days=int(day_offset))

                # Gradually shift to off-hours (the longer the drift, the more off-hours)
                off_hours_prob = 0.1 + (week_i / drift_weeks) * 0.5
                if rng.random() < off_hours_prob:
                    # Off-hours: 21:00–06:00
                    hour = rng.choice([21, 22, 23, 0, 1, 2, 3, 4, 5])
                else:
                    hour = rng.integers(8, 18)

                ts = ts_day.replace(
                    hour=int(hour), minute=rng.integers(0, 60),
                    second=rng.integers(0, 60), microsecond=0,
                )

                # Access from expanding resource set (mix of new and typical)
                if current_new_resources and rng.random() < 0.6:
                    resource = str(rng.choice(current_new_resources))
                else:
                    resource = str(rng.choice(target.typical_resources)) \
                        if target.typical_resources else "api_auth"

                # Uses legitimate credentials and device — hard to distinguish
                event = {
                    "entity_id": target.entity_id,
                    "entity_type": target.entity_type,
                    "timestamp": ts,
                    "source_ip": str(rng.choice(target.source_ips))
                        if target.source_ips else "10.0.0.1",
                    "geo_location": f"{target.home_geos[0]['city']}, {target.home_geos[0]['country']}"
                        if target.home_geos else "Unknown",
                    "resource_accessed": resource,
                    "auth_method": str(rng.choice(list(target.auth_methods.keys()))),
                    "session_duration": round(
                        float(rng.lognormal(target.session_mu, target.session_sigma)), 2
                    ),
                    "command_sequence": ",".join(
                        str(rng.choice(list(target.command_weights.keys())))
                        for _ in range(rng.integers(3, 8))
                    ) if target.command_weights else "ssh,ls,exit",
                    "device_fingerprint": str(rng.choice(target.device_fingerprints))
                        if target.device_fingerprints else "fp_unknown",
                    "label": "ambiguous",
                    "attack_subtype": "insider_drift",
                }
                attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
