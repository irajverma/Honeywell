"""
exfiltration.py — Inject low-and-slow data exfiltration patterns.

Pattern: Slightly elevated session durations combined with access to
data-heavy resources over 5+ consecutive days.  Each individual event
looks borderline normal — the anomaly is only visible in aggregate.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


# Resources that hold significant data (exfiltration targets)
DATA_HEAVY_RESOURCES = [
    "db_customers", "db_orders", "db_employees", "db_financials",
    "s3_data_lake", "s3_backups", "share_executive", "share_finance",
    "share_research", "share_engineering", "db_analytics",
]


def inject_exfiltration(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject low-and-slow exfiltration attack events."""
    rate = config.attack_rates.get("exfiltration", 0.004)
    n_campaigns = max(1, int(len(events_df) * rate / 10))  # each campaign = ~10 events

    attack_events = []

    user_profiles = [p for p in profiles if p.entity_type == "user"]
    if not user_profiles:
        return events_df

    for _ in range(n_campaigns):
        target = rng.choice(user_profiles)

        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        # Campaign spans 5–15 days
        campaign_days = rng.integers(5, 16)
        campaign_start = entity_events["timestamp"].min() + timedelta(
            days=int(rng.integers(0, max(1, config.sim_days - campaign_days)))
        )

        # Exfil commands: mixed with normal to stay under the radar
        exfil_cmds = ["cat", "scp", "tar", "gzip", "curl", "rsync", "base64"]

        for day_i in range(campaign_days):
            ts_day = campaign_start + timedelta(days=day_i)

            # 1–2 events per day (low and slow)
            n_daily = rng.integers(1, 3)
            for _ in range(n_daily):
                hour = rng.uniform(9, 18)  # During work hours to blend in
                ts = ts_day.replace(
                    hour=int(hour), minute=rng.integers(0, 60),
                    second=rng.integers(0, 60), microsecond=0,
                )

                # Slightly elevated session duration (1.5x–3x normal)
                elevated_duration = round(
                    float(rng.lognormal(target.session_mu, target.session_sigma))
                    * rng.uniform(1.5, 3.0), 2
                )

                # Access data-heavy resource
                resource = str(rng.choice(DATA_HEAVY_RESOURCES))

                # Command sequence with exfil-related commands
                n_cmds = rng.integers(4, 9)
                cmd_list = [str(rng.choice(exfil_cmds)) for _ in range(n_cmds)]
                cmd_seq = ",".join(cmd_list)

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
                    "session_duration": elevated_duration,
                    "command_sequence": cmd_seq,
                    "device_fingerprint": str(rng.choice(target.device_fingerprints))
                        if target.device_fingerprints else "fp_unknown",
                    "label": "anomaly",
                    "attack_subtype": "exfiltration",
                }
                attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
