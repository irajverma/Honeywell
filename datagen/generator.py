"""
generator.py — Orchestrator for the synthetic data generator.

Pipeline:
  1. Generate entity behavioral profiles
  2. Sample normal events for all entities across the 60-day window
  3. Inject attack patterns (7 types) into the event stream
  4. Apply concept drift to ~12% of entities during the test period
  5. Add train/test split column
  6. Sort by timestamp, split into events.csv and ground_truth.csv
"""

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from datagen.config import GeneratorConfig
from datagen.profiles import EntityProfile, generate_all_profiles
from datagen.normal_events import generate_all_normal_events
from datagen.attacks import ATTACK_INJECTORS


def _apply_concept_drift(
    events_df: pd.DataFrame,
    profiles: List[EntityProfile],
    config: GeneratorConfig,
    start_date: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Apply legitimate behavioral shifts to a subset of entities during the
    test period.  These are NOT anomalies — they simulate natural concept
    drift (role change, new project, schedule shift) that the model must
    tolerate without over-flagging.

    Changes applied (randomly per entity):
      - Shift login hour peak by ±1–3 hours
      - Add 1 new resource to the entity's typical set
      - Slight change in session duration distribution
    """
    test_start = start_date + timedelta(days=config.sim_days - config.test_days)

    # Select entities to drift
    n_drift = max(1, int(len(profiles) * config.concept_drift_frac))
    drift_indices = rng.choice(len(profiles), size=n_drift, replace=False)
    drift_entity_ids = {profiles[i].entity_id for i in drift_indices}

    print(f"  Applying concept drift to {n_drift} entities: "
          f"{sorted(drift_entity_ids)[:5]}{'...' if n_drift > 5 else ''}")

    # Identify test-period rows for drifted entities
    mask = (
        events_df["entity_id"].isin(drift_entity_ids) &
        (events_df["timestamp"] >= test_start) &
        (events_df["label"] == "normal")  # only modify normal events
    )

    if mask.sum() == 0:
        return events_df

    df = events_df.copy()

    for eidx in drift_indices:
        profile = profiles[eidx]
        entity_mask = mask & (df["entity_id"] == profile.entity_id)
        n_affected = entity_mask.sum()
        if n_affected == 0:
            continue

        # --- Drift type 1: Shift login hours by ±1–3 hours ---
        hour_shift_amount = rng.choice([-3, -2, -1, 1, 2, 3])
        affected_timestamps = df.loc[entity_mask, "timestamp"].copy()
        shifted = affected_timestamps + timedelta(hours=int(hour_shift_amount))
        df.loc[entity_mask, "timestamp"] = shifted

        # --- Drift type 2: Add 1 new resource ---
        unused_resources = [
            r for r in config.resource_pool
            if r not in profile.typical_resources
        ]
        if unused_resources:
            new_resource = str(rng.choice(unused_resources))
            # Replace ~20% of the entity's test events with the new resource
            replace_mask = entity_mask & (rng.random(len(df)) < 0.20)
            df.loc[replace_mask, "resource_accessed"] = new_resource

        # --- Drift type 3: Slightly change session duration ---
        # Multiply by a consistent factor (0.7–1.4)
        duration_factor = rng.uniform(0.7, 1.4)
        df.loc[entity_mask, "session_duration"] = (
            df.loc[entity_mask, "session_duration"] * duration_factor
        ).round(2)

    return df


def generate_dataset(
    config: GeneratorConfig,
    output_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: profiles → events → attacks → drift → split → output.

    Returns:
        (events_df, ground_truth_df)
    """
    rng = np.random.default_rng(config.random_seed)
    start_date = datetime(2026, 5, 1, 0, 0, 0)  # simulation start
    test_start = start_date + timedelta(days=config.sim_days - config.test_days)

    # ── Step 1: Generate profiles ────────────────────────────────
    print("Step 1/6: Generating entity profiles...")
    profiles = generate_all_profiles(config)
    print(f"  Created {len(profiles)} profiles "
          f"({config.num_users}U + {config.num_service_accounts}S + "
          f"{config.num_edge_devices}E)")

    # ── Step 2: Sample normal events ─────────────────────────────
    print("Step 2/6: Sampling normal events...")
    events_df = generate_all_normal_events(profiles, config, start_date)
    print(f"  Generated {len(events_df):,} normal events")

    # ── Step 3: Inject attacks ───────────────────────────────────
    print("Step 3/6: Injecting attack patterns...")
    attack_rng = np.random.default_rng(config.random_seed + 2000)
    for attack_name, injector_fn in ATTACK_INJECTORS:
        pre_count = len(events_df)
        events_df = injector_fn(events_df, profiles, config, attack_rng)
        injected = len(events_df) - pre_count
        print(f"  {attack_name}: +{injected} events")

    # ── Step 4: Apply concept drift in test period ───────────────
    print("Step 4/6: Applying concept drift to test period...")
    drift_rng = np.random.default_rng(config.random_seed + 3000)
    events_df = _apply_concept_drift(events_df, profiles, config, start_date, drift_rng)

    # ── Step 5: Add train/test split column ──────────────────────
    print("Step 5/6: Adding train/test split...")
    events_df["split"] = "train"
    events_df.loc[events_df["timestamp"] >= test_start, "split"] = "test"

    # Sort by timestamp
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    train_count = (events_df["split"] == "train").sum()
    test_count = (events_df["split"] == "test").sum()
    print(f"  Train: {train_count:,} events | Test: {test_count:,} events")

    # ── Step 6: Split and write output ───────────────────────────
    print("Step 6/6: Writing output files...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Ground truth: entity_id, timestamp, label, attack_subtype, split
    ground_truth_df = events_df[
        ["entity_id", "timestamp", "label", "attack_subtype", "split"]
    ].copy()

    # Events: drop label and attack_subtype (kept hidden for inference)
    output_events_df = events_df.drop(columns=["label", "attack_subtype"])

    # Write CSVs
    events_path = out_path / "events.csv"
    gt_path = out_path / "ground_truth.csv"

    output_events_df.to_csv(events_path, index=False)
    ground_truth_df.to_csv(gt_path, index=False)

    print(f"\n  events.csv       -> {events_path}  ({len(output_events_df):,} rows)")
    print(f"  ground_truth.csv -> {gt_path}  ({len(ground_truth_df):,} rows)")

    # ── Summary statistics ───────────────────────────────────────
    _print_summary(events_df, config)

    return output_events_df, ground_truth_df


def _print_summary(events_df: pd.DataFrame, config: GeneratorConfig):
    """Print distribution summaries for manual review."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    total = len(events_df)
    print(f"\nTotal events: {total:,}")
    print(f"Simulation: {config.sim_days} days "
          f"(train: {config.sim_days - config.test_days}d, "
          f"test: {config.test_days}d)")

    # Label distribution
    print("\n-- Label Distribution --")
    label_counts = events_df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label:12s}: {count:>7,}  ({pct:.2f}%)")

    # Attack subtype distribution
    print("\n-- Attack Subtype Distribution --")
    anomalous = events_df[events_df["label"].isin(["anomaly", "ambiguous"])]
    if not anomalous.empty:
        subtype_counts = anomalous["attack_subtype"].value_counts()
        for subtype, count in subtype_counts.items():
            pct = count / total * 100
            print(f"  {subtype:22s}: {count:>5,}  ({pct:.2f}%)")

    # Events per entity type
    print("\n-- Events by Entity Type --")
    etype_counts = events_df["entity_type"].value_counts()
    for etype, count in etype_counts.items():
        print(f"  {etype:18s}: {count:>7,}")

    # Train/test split
    print("\n-- Train/Test Split --")
    split_counts = events_df["split"].value_counts()
    for split, count in split_counts.items():
        pct = count / total * 100
        print(f"  {split:6s}: {count:>7,}  ({pct:.1f}%)")

    # Anomaly rate by split
    print("\n-- Anomaly Rate by Split --")
    for split_val in ["train", "test"]:
        subset = events_df[events_df["split"] == split_val]
        n_anomaly = (subset["label"] == "anomaly").sum()
        n_ambiguous = (subset["label"] == "ambiguous").sum()
        if len(subset) > 0:
            print(f"  {split_val}: anomaly={n_anomaly} ({n_anomaly/len(subset)*100:.2f}%), "
                  f"ambiguous={n_ambiguous} ({n_ambiguous/len(subset)*100:.2f}%)")

    print("\n" + "=" * 60)
