"""
impossible_travel.py — Inject impossible-travel attack patterns.

Pattern: Two logins from geographically distant locations (>5000 km apart)
within a short time gap (<1 hour), which is physically impossible.
"""

import math
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _find_distant_geo(home_geo: dict, config: GeneratorConfig, rng: np.random.Generator,
                       min_distance_km: float = 5000.0) -> dict:
    """Find a geo location at least min_distance_km from home."""
    candidates = [
        g for g in config.geo_locations
        if _haversine_km(home_geo["lat"], home_geo["lon"],
                         g["lat"], g["lon"]) >= min_distance_km
    ]
    if not candidates:
        # Fallback: just pick the farthest one
        candidates = sorted(
            config.geo_locations,
            key=lambda g: _haversine_km(home_geo["lat"], home_geo["lon"],
                                         g["lat"], g["lon"]),
            reverse=True,
        )
    return candidates[rng.integers(min(len(candidates), 3))]


def inject_impossible_travel(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject impossible-travel attack events."""
    rate = config.attack_rates.get("impossible_travel", 0.003)
    n_attacks = max(1, int(len(events_df) * rate))

    attack_events = []
    user_profiles = [p for p in profiles if p.entity_type == "user"]
    if not user_profiles:
        return events_df

    for _ in range(n_attacks):
        target = rng.choice(user_profiles)
        entity_events = events_df[events_df["entity_id"] == target.entity_id]
        if entity_events.empty:
            continue

        # Pick a legitimate event as the "first login"
        base_event = entity_events.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        base_ts = base_event["timestamp"]

        # Home geo from the legitimate event
        home_geo = target.home_geos[0]

        # Find a distant location
        distant_geo = _find_distant_geo(home_geo, config, rng)

        # Second login: 5–55 minutes later (impossibly fast)
        time_gap = timedelta(minutes=int(rng.integers(5, 56)))
        travel_ts = base_ts + time_gap

        # Generate the anomalous "traveled" login
        travel_ip = f"78.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"

        event = {
            "entity_id": target.entity_id,
            "entity_type": target.entity_type,
            "timestamp": travel_ts,
            "source_ip": travel_ip,
            "geo_location": f"{distant_geo['city']}, {distant_geo['country']}",
            "resource_accessed": str(rng.choice(target.typical_resources))
                if target.typical_resources else "api_auth",
            "auth_method": str(rng.choice(list(target.auth_methods.keys()))),
            "session_duration": round(float(rng.lognormal(target.session_mu, target.session_sigma)), 2),
            "command_sequence": "ssh,ls,cat,scp,exit",
            "device_fingerprint": f"fp_travel_{rng.integers(1000,9999)}",
            "label": "anomaly",
            "attack_subtype": "impossible_travel",
        }
        attack_events.append(event)

    if attack_events:
        attack_df = pd.DataFrame(attack_events)
        attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
        events_df = pd.concat([events_df, attack_df], ignore_index=True)

    return events_df
