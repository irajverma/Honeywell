"""
run_baseline.py — CLI entry point for Phase 2 Baseline Statistical Profiling Model.

Usage:
    python run_baseline.py
    python run_baseline.py --min-events 10 --data-dir data/
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.baseline_profiler import BaselineProfiler


def verify_cold_start_fallback(events_df: pd.DataFrame, min_events: int = 10):
    """
    Verification test (Option b): Artificially truncate a few entities' training history
    to <min_events and confirm that population fallback profiles are cleanly retrieved.
    """
    print(f"\n{'-'*60}")
    print("Running Cold-Start Verification Test (Option b)...")
    train_df = events_df[events_df["split"] == "train"].copy()

    # Select 5 distinct entities across available types
    unique_entities = train_df[["entity_id", "entity_type"]].drop_duplicates()
    sample_entities = unique_entities.sample(min(5, len(unique_entities)), random_state=42)

    # Artificially truncate their training history to 3 events (< min_events)
    truncated_dfs = []
    sparse_ids = set(sample_entities["entity_id"])

    for eid in sparse_ids:
        entity_rows = train_df[train_df["entity_id"] == eid].head(3)
        truncated_dfs.append(entity_rows)

    # Keep all other entities untouched
    other_rows = train_df[~train_df["entity_id"].isin(sparse_ids)]
    test_train_df = pd.concat([other_rows] + truncated_dfs, ignore_index=True)

    # Fit a verification profiler
    test_profiler = BaselineProfiler(min_events=min_events)
    test_profiler.fit(test_train_df)

    # Assert and verify fallback behavior
    all_passed = True
    for _, row in sample_entities.iterrows():
        eid = str(row["entity_id"])
        etype = str(row["entity_type"])
        profile = test_profiler.get_profile(eid, etype)

        if not profile.is_population_fallback:
            print(f"  [FAIL] Entity {eid} ({etype}) with 3 events did NOT trigger fallback!")
            all_passed = False
        elif profile.entity_id != f"POPULATION_{etype.upper()}":
            print(f"  [FAIL] Entity {eid} retrieved wrong profile ID: {profile.entity_id}")
            all_passed = False
        else:
            print(f"  [OK] Checked {eid} ({etype}, 3 train events) -> Retrieved {profile.entity_id} (fallback=True)")

    if all_passed:
        print(f"[PASS] Cold-Start Verification Passed: All 5 sparse entities (<{min_events} events) cleanly retrieved population fallback profiles!")
    else:
        raise RuntimeError("Cold-start verification failed!")
    print(f"{'-'*60}\n")


def print_evaluation_table(results: dict):
    """Print a clean Markdown table of evaluation metrics."""
    print("\n# Baseline Model Evaluation Results (Test Split)\n")
    print("| Metric / Setting | Core Anomaly (`anomaly`) | Insider Drift (`ambiguous`) |")
    print("|---|---|---|")
    
    anom = results["anomaly"]
    ambig = results["ambiguous"]

    print(f"| **PR-AUC (Primary Metric)** | **{anom['pr_auc']:.4f}** | **{ambig['pr_auc']:.4f}** |")
    print(f"| Support (Positive / Negative) | {anom['support_pos']:,} / {anom['support_neg']:,} | {ambig['support_pos']:,} / {ambig['support_neg']:,} |")
    print("| --- | --- | --- |")
    print(f"| Precision (Score >= 2.0) | {anom['prec_thresh_2.0']:.4f} | {ambig['prec_thresh_2.0']:.4f} |")
    print(f"| Recall (Score >= 2.0)    | {anom['rec_thresh_2.0']:.4f} | {ambig['rec_thresh_2.0']:.4f} |")
    print(f"| F1-Score (Score >= 2.0)  | {anom['f1_thresh_2.0']:.4f} | {ambig['f1_thresh_2.0']:.4f} |")
    print("| --- | --- | --- |")
    print(f"| Precision (Top 1% Budget)| {anom['prec_top_1pct']:.4f} | {ambig['prec_top_1pct']:.4f} |")
    print(f"| Recall (Top 1% Budget)   | {anom['rec_top_1pct']:.4f} | {ambig['rec_top_1pct']:.4f} |")
    print(f"| F1-Score (Top 1% Budget) | {anom['f1_top_1pct']:.4f} | {ambig['f1_top_1pct']:.4f} |\n")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Baseline Statistical Profiler")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing events.csv and ground_truth.csv")
    parser.add_argument("--min-events", type=int, default=10, help="Minimum training events for individual profile before population fallback")
    parser.add_argument("--output", type=str, default="data/baseline_scores.csv", help="Output path for scored CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    events_path = data_dir / "events.csv"
    gt_path = data_dir / "ground_truth.csv"

    if not events_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing required data files in {data_dir}/. Ensure Phase 1 generator has run.")

    print(f"\n{'='*60}")
    print("Phase 2: Baseline Statistical Profiling Model")
    print(f"{'='*60}")
    print(f"Loading data from {data_dir}/...")
    start_load = time.time()
    events_df = pd.read_csv(events_path)
    gt_df = pd.read_csv(gt_path)
    print(f"  Loaded {len(events_df):,} events in {time.time() - start_load:.2f}s")

    # Run cold-start verification test (Option b)
    verify_cold_start_fallback(events_df, min_events=args.min_events)

    # Fit profiler on real training split
    print("Fitting BaselineProfiler on training split...")
    start_fit = time.time()
    profiler = BaselineProfiler(min_events=args.min_events)
    profiler.fit(events_df)
    print(f"  Fitting complete in {time.time() - start_fit:.2f}s")

    # Score all events
    print("Scoring event stream...")
    start_score = time.time()
    scored_df = profiler.score_dataframe(events_df)
    print(f"  Scoring complete in {time.time() - start_score:.2f}s")

    # Evaluate on test split
    print("Evaluating baseline on test split against ground truth...")
    test_scored = scored_df[scored_df["split"] == "test"].copy()
    test_gt = gt_df[gt_df["split"] == "test"].copy()
    results = profiler.evaluate(test_scored, test_gt)

    print_evaluation_table(results)

    # Save output CSV: (entity_id, timestamp, baseline_score, flagged_reasons)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = scored_df[["entity_id", "timestamp", "baseline_score", "flagged_reasons"]]
    out_df.to_csv(out_path, index=False)
    print(f"Saved scored output ({len(out_df):,} rows) to {out_path}\n")


if __name__ == "__main__":
    main()
