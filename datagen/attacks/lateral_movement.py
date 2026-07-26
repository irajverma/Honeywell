"""
lateral_movement.py — Inject lateral-movement attack patterns.

Pattern: An entity accesses 3+ resources outside their typical resource set
in a sequential chain within a short time window, suggesting an attacker
pivoting through the network after initial compromise.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def inject_lateral_movement(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject lateral-movement attack events."""
    rate = config.attack_rates.get("lateral_movement", 0.003)
    n_target_events = max(1, int(len(events_df) * rate))
    avg_hops_per_campaign = 5  # 3-6 hops
    n_attacks = max(1, n_target_events // avg_hops_per_campaign)

    attack_events = []

    # Any entity type can be a pivot point
    for _ in range(n_attacks):
        target = profiles[rng.integers(len(profiles))]

        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        base_event = entity_events.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        base_ts = base_event["timestamp"]

        # Resources outside the entity's typical set
        unusual_resources = [
            r for r in config.resource_pool
            if r not in target.typical_resources
        ]
        if len(unusual_resources) < 3:
            continue

        # Access 3–6 unusual resources in sequence
        n_hops = rng.integers(3, 7)
        hop_resources = [
            unusual_resources[i]
            for i in rng.choice(len(unusual_resources), size=min(n_hops, len(unusual_resources)), replace=False)
        ]

        # Lateral movement commands
        lateral_cmds = ["ssh", "net_use", "psexec", "wmic", "mimikatz",
                        "pass_the_hash", "rdp", "powershell", "whoami",
                        "ipconfig", "net_view"]

        for hop_i, resource in enumerate(hop_resources):
            ts = base_ts + timedelta(minutes=int(hop_i * rng.integers(2, 15)))

            # Mix of normal and suspicious commands
            n_cmds = rng.integers(3, 7)
            cmd_seq = ",".join(
                str(rng.choice(lateral_cmds)) for _ in range(n_cmds)
            )

            event = {
                "entity_id": target.entity_id,
                "entity_type": target.entity_type,
                "timestamp": ts,
                "source_ip": str(rng.choice(target.source_ips)) if target.source_ips
                    else f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}",
                "geo_location": f"{target.home_geos[0]['city']}, {target.home_geos[0]['country']}"
                    if target.home_geos else "Unknown",
                "resource_accessed": resource,
                "auth_method": str(rng.choice(list(target.auth_methods.keys()))),
                "session_duration": round(float(rng.lognormal(1.5, 0.5)), 2),
                "command_sequence": cmd_seq,
                "device_fingerprint": str(rng.choice(target.device_fingerprints))
                    if target.device_fingerprints else "fp_unknown",
                "label": "anomaly",
                "attack_subtype": "lateral_movement",
            }
            attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
