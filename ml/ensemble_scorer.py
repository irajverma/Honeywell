"""
ensemble_scorer.py — Hybrid Ensemble Risk Scorer and SOC Dashboard Feed Generator (Phase 4).

Combines Phase 2 static profiling rules with Phase 3 deep learning sequence models
using a Tiered SOC Prioritization Architecture that preserves 100% of deterministic static
detections while leveraging sequence modeling to capture complex multi-step anomalies.
Evaluates 4-way performance and exports pre-aggregated JSON for the Cyber SOC Dashboard.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ml.evaluation import evaluate_model_performance, print_comparison_table, ALL_ATTACK_SUBTYPES


def compute_hybrid_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement a Tiered SOC Prioritization Architecture that prevents continuous
    unsupervised ML noise from locking out high-precision deterministic static rules.
    """
    out_df = df.copy()
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])
    out_df = out_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # 1. Compute recent distinct resources accessed per entity (last 10 events / ~24h equivalent)
    res_counts = []
    for _, group in out_df.groupby("entity_id", sort=False):
        res_series = group["resource_accessed"].astype(str).to_list()
        counts = [len(set(res_series[max(0, i - 9): i + 1])) for i in range(len(res_series))]
        res_counts.extend(counts)
    out_df["distinct_resources_recent"] = res_counts

    # 2. Map individual model scores to a [0, 100] percentile risk scale
    for col in ["baseline_score", "iforest_score", "lstm_score"]:
        risk_col = col.replace("_score", "_risk")
        out_df[risk_col] = out_df[col].rank(pct=True) * 100.0

    # 3. Tier-Gated Monotonic Hybrid Ensemble Architecture:
    # High-confidence deterministic rule violations (baseline_score >= 3.0) are preserved intact at top priority.
    # For baseline_score < 3.0, an additive sequence boost (0.99 * lstm_pct) is applied to entities exhibiting
    # resource traversal (distinct_resources_recent >= 5), ensuring zero crowding out at 1.0% budget while
    # elevating sequential lateral movement and exfiltration anomalies as budget expands.
    lstm_pct = out_df["lstm_score"].rank(pct=True)
    is_high_rule = out_df["baseline_score"] >= 3.0
    is_resource_chain = out_df["distinct_resources_recent"] >= 5
    sequence_boost = np.where(is_resource_chain, 0.99 * lstm_pct, 0.0)

    out_df["hybrid_score"] = np.where(
        is_high_rule,
        out_df["baseline_score"],
        out_df["baseline_score"] + sequence_boost
    )

    # Map hybrid_score to a readable [0, 100] percentile risk scale for dashboard display
    out_df["hybrid_risk"] = out_df["hybrid_score"].rank(pct=True) * 100.0

    # 4. Assign Severity Badges based on Hybrid Risk percentile
    conds = [
        out_df["hybrid_risk"] >= 99.0, # Top 1% -> Critical
        out_df["hybrid_risk"] >= 95.0, # Top 5% -> High
        out_df["hybrid_risk"] >= 85.0, # Top 15% -> Medium
    ]
    choices = ["Critical", "High", "Medium"]
    out_df["severity"] = np.select(conds, choices, default="Low")

    # 5. Generate Automated SOC Incident Narratives
    explanations = []
    for row in out_df.itertuples():
        if row.baseline_score >= 3.0:
            exp = (f"Deterministic Policy Violation: Unusual access from {row.geo_location} via {row.device_fingerprint} "
                   f"targeting {row.resource_accessed}. Static Rule Score: {row.baseline_score:.1f}/5.5.")
        elif row.distinct_resources_recent >= 5 and row.lstm_risk >= 95.0:
            exp = (f"Sequence Traversal Anomaly Alert: Entity {row.entity_id} accessed {row.distinct_resources_recent} distinct "
                   f"resources in rapid sequence. LSTM Sequence Risk: {row.lstm_risk:.1f}/100.")
        elif row.iforest_risk >= 95.0:
            exp = (f"Tabular Distribution Outlier: Unusual combination of session duration, time delta, and auth method. "
                   f"Tabular ML Risk: {row.iforest_risk:.1f}/100.")
        else:
            exp = (f"Behavioral Anomaly Detected: Combined Hybrid Risk Score {row.hybrid_risk:.1f}/100 "
                   f"(Baseline: {row.baseline_risk:.1f}, LSTM: {row.lstm_risk:.1f}).")
        explanations.append(exp)
    
    out_df["explanation"] = explanations
    return out_df


def generate_dashboard_feed(df: pd.DataFrame, eval_results: dict, output_path: str = "data/dashboard_feed.json"):
    """Export pre-aggregated KPIs, model comparison metrics, alerts, and timeline for Dashboard rendering."""
    print(f"Generating Cyber SOC Dashboard feed -> {output_path}...")
    
    hybrid_eval = eval_results["Hybrid Ensemble (Monotonic Blend)"]["anomaly"]
    kpis = {
        "total_events": int(len(df)),
        "total_alerts_top1pct": int(len(df[df["hybrid_risk"] >= 99.0])),
        "critical_threats": int(len(df[df["severity"] == "Critical"])),
        "precision": float(hybrid_eval["precision"]),
        "recall": float(hybrid_eval["recall"]),
        "f1": float(hybrid_eval["f1"]),
        "pr_auc": float(hybrid_eval["pr_auc"]),
    }

    model_comp = {}
    for model_name, res in eval_results.items():
        model_comp[model_name] = {
            "pr_auc": res["anomaly"]["pr_auc"],
            "recall": res["anomaly"]["recall"],
            "precision": res["anomaly"]["precision"],
            "f1": res["anomaly"]["f1"],
            "per_attack_recall": {k: v["recall"] for k, v in res["per_attack_recall"].items()}
        }

    top_alerts_df = df.sort_values("hybrid_score", ascending=False).head(1000)
    alerts_list = []
    for row in top_alerts_df.itertuples():
        alerts_list.append({
            "timestamp": str(row.timestamp),
            "entity_id": str(row.entity_id),
            "entity_type": str(row.entity_type),
            "attack_subtype": str(row.attack_subtype) if pd.notna(row.attack_subtype) and str(row.attack_subtype) != "nan" else "normal",
            "label": str(row.label),
            "hybrid_risk": round(float(row.hybrid_risk), 1),
            "baseline_risk": round(float(row.baseline_risk), 1),
            "lstm_risk": round(float(row.lstm_risk), 1),
            "severity": str(row.severity),
            "explanation": str(row.explanation),
            "geo_location": str(row.geo_location),
            "resource_accessed": str(row.resource_accessed),
            "device_fingerprint": str(row.device_fingerprint),
        })

    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    daily_total = df.groupby("day").size().to_dict()
    daily_alerts = df[df["hybrid_risk"] >= 99.0].groupby("day").size().to_dict()
    
    timeline_days = sorted(list(daily_total.keys()))
    timeline_data = {
        "days": timeline_days,
        "total_events": [daily_total.get(d, 0) for d in timeline_days],
        "alerts": [daily_alerts.get(d, 0) for d in timeline_days],
    }

    feed = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "kpis": kpis,
        "model_comparison": model_comp,
        "alerts": alerts_list,
        "timeline": timeline_data,
        "attack_subtypes": ALL_ATTACK_SUBTYPES,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)
    print(f"  Dashboard feed successfully exported ({len(alerts_list)} top alerts, {len(timeline_days)} days timeline).")


def main():
    print(f"\n{'='*70}")
    print("Phase 4: Hybrid Ensemble Risk Scoring & SOC Dashboard Feed Generation")
    print(f"{'='*70}")

    scores_path = Path("data/phase3_scores.csv")
    events_path = Path("data/events.csv")
    gt_path = Path("data/ground_truth.csv")

    if not scores_path.exists() or not events_path.exists() or not gt_path.exists():
        raise FileNotFoundError("Missing data/ files. Ensure run_phase3.py has been executed.")

    print("Loading datasets...")
    scores_df = pd.read_csv(scores_path)
    events_df = pd.read_csv(events_path).sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    gt_df = pd.read_csv(gt_path).sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    if not (len(scores_df) == len(events_df) == len(gt_df)):
        raise ValueError(f"Row count mismatch: scores={len(scores_df)}, events={len(events_df)}, gt={len(gt_df)}")


    df_hybrid = scores_df.copy()
    df_hybrid["split"] = gt_df["split"]
    df_hybrid["label"] = gt_df["label"]
    df_hybrid["attack_subtype"] = gt_df["attack_subtype"]
    df_hybrid["resource_accessed"] = events_df["resource_accessed"]
    df_hybrid["geo_location"] = events_df["geo_location"]
    df_hybrid["device_fingerprint"] = events_df["device_fingerprint"]
    df_hybrid["entity_type"] = events_df["entity_type"]

    print(f"  Loaded {len(df_hybrid):,} events. Computing Hybrid Ensemble risk scores...")
    df_hybrid = compute_hybrid_risk_scores(df_hybrid)

    # Evaluate 4-way comparison on test split
    print(f"\n{'='*70}")
    print("Evaluating 4-Way Model Comparison on Test Split (@ Top 1% Alert Budget)...")
    print(f"{'='*70}")

    test_sub = df_hybrid[df_hybrid["split"] == "test"].copy()
    test_scored = test_sub.drop(columns=["label", "attack_subtype"])
    test_gt = gt_df[gt_df["split"] == "test"].copy()
    
    models_to_eval = {
        "Baseline Profiler (Static Rules)": "baseline_score",
        "Isolation Forest (Tabular ML)": "iforest_score",
        "LSTM Autoencoder (Sequence ML)": "lstm_score",
        "Hybrid Ensemble (Monotonic Blend)": "hybrid_score",
    }

    eval_results = {}
    for name, col in models_to_eval.items():
        eval_results[name] = evaluate_model_performance(test_scored, test_gt, score_col=col, top_k_pct=0.01)

    print_comparison_table(eval_results, top_k_pct=0.01)

    # Generate dashboard feed
    generate_dashboard_feed(df_hybrid, eval_results, output_path="data/dashboard_feed.json")
    print(f"\nPhase 4 hybrid ensemble scoring and data export complete!\n")


if __name__ == "__main__":
    main()
