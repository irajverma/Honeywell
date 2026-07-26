"""
credential_stuffing.py — Inject credential-stuffing attack patterns.

Pattern: A small set of attacker IPs attempt logins against many distinct
entities with low success rates (~5–15%).  Unlike brute force, this targets
breadth across entities rather than depth against one.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def inject_credential_stuffing(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject credential-stuffing attack events."""
    rate = config.attack_rates.get("credential_stuffing", 0.004)
    n_total_events = max(1, int(len(events_df) * rate))

    attack_events = []

    # Create a small attacker IP pool (botnet)
    n_attacker_ips = rng.integers(2, 6)
    attacker_ips = [
        f"103.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"
        for _ in range(n_attacker_ips)
    ]

    # Attacker geo (typically a single region)
    attacker_geo_idx = rng.integers(len(config.geo_locations))
    attacker_geo = config.geo_locations[attacker_geo_idx]
    attacker_geo_str = f"{attacker_geo['city']}, {attacker_geo['country']}"

    # Target many distinct user entities
    user_profiles = [p for p in profiles if p.entity_type == "user"]
    if not user_profiles:
        return events_df

    # Pick 3 time windows for campaigns: early train (~day 10), late train (~day 35), and test split (~day 52)
    all_timestamps = events_df["timestamp"].dropna()
    if all_timestamps.empty:
        return events_df

    min_ts = all_timestamps.min()
    max_ts = all_timestamps.max()
    total_days = max(1, (max_ts - min_ts).days)

    # 3 campaign start times guaranteeing coverage in both train and test splits
    campaign_starts = [
        min_ts + timedelta(days=int(rng.uniform(2, min(20, total_days * 0.35)))),
        min_ts + timedelta(days=int(rng.uniform(25, min(40, total_days * 0.70)))),
        min_ts + timedelta(days=int(rng.uniform(int(total_days * 0.82), max(int(total_days * 0.82) + 1, total_days - 2)))),
    ]

    events_per_campaign = max(1, n_total_events // len(campaign_starts))

    for camp_idx, campaign_start in enumerate(campaign_starts):
        campaign_duration_hours = rng.integers(2, 8)

        # Spread attempts across many entities for this campaign
        n_targets = min(len(user_profiles), rng.integers(8, 25))
        targets = [user_profiles[i] for i in rng.choice(len(user_profiles), size=n_targets, replace=False)]

        events_per_target = max(1, events_per_campaign // n_targets)
        success_rate = rng.uniform(0.05, 0.15)

        for target in targets:
            for attempt_i in range(events_per_target):
                ts = campaign_start + timedelta(
                    seconds=int(rng.uniform(0, campaign_duration_hours * 3600))
                )
                is_success = rng.random() < success_rate

                event = {
                    "entity_id": target.entity_id,
                    "entity_type": target.entity_type,
                    "timestamp": ts,
                    "source_ip": str(rng.choice(attacker_ips)),
                    "geo_location": attacker_geo_str,
                    "resource_accessed": "api_auth",
                    "auth_method": "password",
                    "session_duration": round(float(rng.uniform(0.5, 3.0) if is_success
                                                    else rng.uniform(0.01, 0.1)), 2),
                    "command_sequence": "login_attempt,success,ls,exit" if is_success
                                        else "login_attempt,fail",
                    "device_fingerprint": f"fp_stuff_{rng.integers(100,999)}",
                    "label": "anomaly",
                    "attack_subtype": "credential_stuffing",
                }
                attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
