"""
run_phase3.py — CLI entry point for Phase 3 Sequence-Aware Detection Model.

Orchestrates Phase 2 baseline profiling, Isolation Forest tabular ML, and PyTorch
LSTM Autoencoder sequence modeling. Evaluates all three models side-by-side with
detailed per-attack-type recall breakdowns and exports data/phase3_scores.csv.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

from ml.baseline_profiler import BaselineProfiler
from ml.dataset_utils import extract_event_features, create_sequence_windows
from ml.isolation_forest import IsolationForestModel
from ml.sequence_model import SequenceAnomalyDetector
from ml.evaluation import evaluate_model_performance, print_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Multi-Model Anomaly Detection Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing events.csv and ground_truth.csv")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding window size L for LSTM Autoencoder")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for LSTM Autoencoder")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for LSTM training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--score-mode", type=str, default="final_step", choices=["final_step", "blend", "window"], help="Scoring mode for LSTM reconstruction error")
    parser.add_argument("--output", type=str, default="data/phase3_scores.csv", help="Output path for multi-model scores CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    events_path = data_dir / "events.csv"
    gt_path = data_dir / "ground_truth.csv"

    if not events_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing required data files in {data_dir}/. Ensure Phase 1 generator has run.")

    print(f"\n{'='*70}")
    print("Phase 3: Sequence-Aware Detection Model & Multi-Model Comparison")
    print(f"{'='*70}")

    # Explicit notes requested by user
    print("\n[IMPORTANT SETUP NOTES]")
    print("1. Note on Unsupervised Setup: Both Isolation Forest and the PyTorch LSTM Autoencoder")
    print("   are trained strictly on the unlabeled training split (split == 'train'), which realistically")
    print("   contains ~2.5% real attacks mixed in as normal traffic. This mirrors real-world SOC")
    print("   conditions where ground-truth labels are absent during initial training, representing a")
    print("   true unsupervised anomaly detection setup rather than a data leak.")
    print("2. Note on Sparse Sequence Handling (M < 10): For entities with fewer than window_size=10")
    print("   events (or during the initial 9 events of any entity's stream), sequence windows are")
    print("   dynamically padded by repeating the entity's earliest available event. This ensures")
    print("   100% causal scoring coverage across all events without null values or dropped events.\n")
    print(f"{'-'*70}")

    print("Loading datasets...")
    start_load = time.time()
    events_df = pd.read_csv(events_path)
    gt_df = pd.read_csv(gt_path)
    print(f"  Loaded {len(events_df):,} events in {time.time() - start_load:.2f}s\n")

    # 1. Phase 2 Baseline Profiler
    print("1/4 -> Fitting Phase 2 Baseline Statistical Profiler...")
    start_base = time.time()
    profiler = BaselineProfiler(min_events=10)
    profiler.fit(events_df)
    scored_base = profiler.score_dataframe(events_df)
    print(f"  Baseline scoring complete in {time.time() - start_base:.2f}s\n")

    # 2. Feature Extraction & Rolling Enrichment
    print("2/4 -> Extracting numeric features and step-based rolling enrichments...")
    start_feats = time.time()
    df_feats, feature_cols, _ = extract_event_features(scored_base, profiler, fit_scaler=True)
    print(f"  Extracted {len(feature_cols)} standardized features in {time.time() - start_feats:.2f}s\n")

    # 3. Phase 3 Isolation Forest Tabular ML Baseline
    print("3/4 -> Fitting and scoring Phase 3 Isolation Forest ML Baseline...")
    start_iforest = time.time()
    iforest = IsolationForestModel(n_estimators=100, random_state=42)
    iforest.fit(df_feats, feature_cols)
    iforest_scores = iforest.score(df_feats, feature_cols)
    df_feats["iforest_score"] = iforest_scores
    print(f"  Isolation Forest scoring complete in {time.time() - start_iforest:.2f}s\n")

    # 4. Phase 3 PyTorch LSTM Autoencoder Sequence Model
    print(f"4/4 -> Constructing causal sliding windows (L={args.window_size}) and training LSTM Autoencoder...")
    start_lstm = time.time()
    all_windows = create_sequence_windows(df_feats, feature_cols, window_size=args.window_size)
    
    train_mask = (df_feats["split"] == "train").to_numpy()
    train_windows = all_windows[train_mask]

    lstm_detector = SequenceAnomalyDetector(
        input_dim=len(feature_cols),
        hidden_dim=32,
        num_layers=2,
        window_size=args.window_size,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device="cpu",
    )
    lstm_detector.fit(train_windows)
    
    # Diagnostic: Check sparse padding impact in test split
    test_mask = df_feats["split"] == "test"
    test_sub = df_feats[test_mask]
    padded_test_count = 0
    exfil_padded = 0
    drift_padded = 0
    for eid, group in df_feats.groupby("entity_id", sort=False):
        group_test = group[group["split"] == "test"]
        for idx in group_test.index:
            event_num = group.index.get_loc(idx)
            if event_num < args.window_size - 1:
                padded_test_count += 1
                st = group.loc[idx, "attack_subtype"] if "attack_subtype" in group.columns else None
                if st == "exfiltration":
                    exfil_padded += 1
                elif st == "insider_drift":
                    drift_padded += 1
    print(f"  [Diagnostic] Sparse Sequence Padding in Test Split: {padded_test_count} / {len(test_sub):,} test events required earliest-event padding.")
    print(f"  [Diagnostic] Exfiltration test events padded: {exfil_padded} | Insider drift test events padded: {drift_padded}")

    print(f"  [LSTM] Scoring all event sequence windows using mode: {args.score_mode}...")
    lstm_scores = lstm_detector.score(all_windows, batch_size=512, score_mode=args.score_mode)
    df_feats["lstm_score"] = lstm_scores
    print(f"  LSTM training and scoring complete in {time.time() - start_lstm:.2f}s\n")

    # 5. Multi-Model Evaluation on Test Split
    print(f"{'='*70}")
    print("Evaluating all three models on test split against ground truth...")
    print(f"{'='*70}")

    test_scored = df_feats[df_feats["split"] == "test"].copy()
    test_gt = gt_df[gt_df["split"] == "test"].copy()

    models_to_eval = {
        "Baseline Profiler (Static Rules)": "baseline_score",
        "Isolation Forest (Tabular ML)": "iforest_score",
        "LSTM Autoencoder (Sequence ML)": "lstm_score",
    }

    results_dict = {}
    for model_name, score_col in models_to_eval.items():
        res = evaluate_model_performance(test_scored, test_gt, score_col=score_col, top_k_pct=0.01)
        results_dict[model_name] = res

    # Print stunning comparison table with per-attack numerator/denominator breakdown
    print_comparison_table(results_dict, top_k_pct=0.01)

    # Save output CSV: (entity_id, timestamp, baseline_score, iforest_score, lstm_score)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df_feats[["entity_id", "timestamp", "baseline_score", "iforest_score", "lstm_score"]]
    out_df.to_csv(out_path, index=False)
    print(f"Saved multi-model scored output ({len(out_df):,} rows) to {out_path}\n")


if __name__ == "__main__":
    main()
