"""
device_spoofing.py — Inject device-spoofing attack patterns.

Pattern: The device fingerprint changes mid-session or a login comes from
a fingerprint never seen in the entity's known device set, suggesting
the attacker is impersonating a trusted device.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def inject_device_spoofing(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject device-spoofing attack events."""
    rate = config.attack_rates.get("device_spoofing", 0.003)
    n_target_events = max(1, int(len(events_df) * rate))
    avg_events_per_campaign = 2  # 1-3 events per spoofed session
    n_attacks = max(1, n_target_events // avg_events_per_campaign)

    attack_events = []

    for _ in range(n_attacks):
        target = profiles[rng.integers(len(profiles))]

        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        base_event = entity_events.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        base_ts = base_event["timestamp"]

        # Generate a spoofed fingerprint that doesn't match any known device
        spoofed_fp = f"fp_spoof_{rng.integers(10000, 99999)}"

        # The spoofed session may look otherwise normal — the key signal
        # is the unknown device fingerprint
        resource = str(rng.choice(target.typical_resources)) if target.typical_resources else "api_auth"
        auth = str(rng.choice(list(target.auth_methods.keys())))

        # Sometimes the attacker also comes from an unusual IP
        if rng.random() < 0.4:
            source_ip = f"91.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"
        else:
            source_ip = str(rng.choice(target.source_ips)) if target.source_ips else "10.0.0.1"

        geo = target.home_geos[0] if target.home_geos else config.geo_locations[0]
        geo_str = f"{geo['city']}, {geo['country']}"

        # Generate 1–3 events with the spoofed device
        n_events = rng.integers(1, 4)
        for ev_i in range(n_events):
            ts = base_ts + timedelta(minutes=int(ev_i * rng.integers(1, 10)))

            cmds = list(target.command_weights.keys()) if target.command_weights else ["ssh", "ls", "exit"]
            n_cmds = rng.integers(3, 7)
            cmd_seq = ",".join(str(rng.choice(cmds)) for _ in range(n_cmds))

            event = {
                "entity_id": target.entity_id,
                "entity_type": target.entity_type,
                "timestamp": ts,
                "source_ip": source_ip,
                "geo_location": geo_str,
                "resource_accessed": resource,
                "auth_method": auth,
                "session_duration": round(float(rng.lognormal(target.session_mu, target.session_sigma)), 2),
                "command_sequence": cmd_seq,
                "device_fingerprint": spoofed_fp,
                "label": "anomaly",
                "attack_subtype": "device_spoofing",
            }
            attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
