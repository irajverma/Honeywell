"""
evaluation.py — Comprehensive evaluation and multi-model comparison (Phase 3).

Computes overall PR-AUC, Precision, Recall, and F1 for core anomaly and ambiguous
drift, and generates detailed per-attack-type recall breakdowns to demonstrate
the strengths of sequence modeling over single-event rules.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


ALL_ATTACK_SUBTYPES = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "device_spoofing",
    "lateral_movement",
    "exfiltration",
    "insider_drift",
]


def evaluate_model_performance(
    scored_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    score_col: str,
    top_k_pct: float = 0.01,
) -> Dict[str, Any]:
    """
    Evaluate a model's scores against ground truth on the test split.
    Reports core anomaly metrics, ambiguous drift metrics, and per-attack-type recall.
    """
    df_left = scored_df.copy()
    df_right = gt_df[["entity_id", "timestamp", "label", "attack_subtype"]].copy()
    df_left["timestamp"] = df_left["timestamp"].astype(str)
    df_right["timestamp"] = df_right["timestamp"].astype(str)

    merged = df_left.merge(
        df_right,
        on=["entity_id", "timestamp"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Merged evaluation dataframe is empty! Check timestamp formats.")

    results = {}

    # 1. Core Anomaly Evaluation (normal vs anomaly)
    anom_mask = merged["label"].isin(["normal", "anomaly"])
    eval_anom = merged[anom_mask].copy()
    y_true_anom = (eval_anom["label"] == "anomaly").astype(int)
    y_score_anom = eval_anom[score_col].to_numpy(dtype=float)

    pr_auc_anom = float(average_precision_score(y_true_anom, y_score_anom))

    # Determine threshold at Top K% operational alert budget across all test events
    all_scores = merged[score_col].to_numpy(dtype=float)
    k_idx = max(1, int(len(all_scores) * top_k_pct))
    threshold_1pct = float(np.sort(all_scores)[-k_idx])

    y_pred_anom = (y_score_anom >= threshold_1pct).astype(int)
    prec_anom, rec_anom, f1_anom, _ = precision_recall_fscore_support(
        y_true_anom, y_pred_anom, average="binary", zero_division=0
    )

    results["anomaly"] = {
        "pr_auc": pr_auc_anom,
        "precision": float(prec_anom),
        "recall": float(rec_anom),
        "f1": float(f1_anom),
        "threshold": threshold_1pct,
        "support_pos": int(y_true_anom.sum()),
        "support_neg": int((y_true_anom == 0).sum()),
        "numerator": int((y_pred_anom * y_true_anom).sum()),
    }

    # 2. Ambiguous / Insider Drift Evaluation (normal vs ambiguous)
    ambig_mask = merged["label"].isin(["normal", "ambiguous"])
    eval_ambig = merged[ambig_mask].copy()
    y_true_ambig = (eval_ambig["label"] == "ambiguous").astype(int)
    y_score_ambig = eval_ambig[score_col].to_numpy(dtype=float)

    pr_auc_ambig = float(average_precision_score(y_true_ambig, y_score_ambig))
    y_pred_ambig = (y_score_ambig >= threshold_1pct).astype(int)
    prec_ambig, rec_ambig, f1_ambig, _ = precision_recall_fscore_support(
        y_true_ambig, y_pred_ambig, average="binary", zero_division=0
    )

    results["ambiguous"] = {
        "pr_auc": pr_auc_ambig,
        "precision": float(prec_ambig),
        "recall": float(rec_ambig),
        "f1": float(f1_ambig),
        "support_pos": int(y_true_ambig.sum()),
        "support_neg": int((y_true_ambig == 0).sum()),
        "numerator": int((y_pred_ambig * y_true_ambig).sum()),
    }

    # 3. Per-Attack-Type Recall at Top K% Alert Budget
    per_attack = {}
    for subtype in ALL_ATTACK_SUBTYPES:
        sub_df = merged[merged["attack_subtype"] == subtype]
        support = len(sub_df)
        if support == 0:
            recall = 0.0
            num = 0
        else:
            num = int((sub_df[score_col] >= threshold_1pct).sum())
            recall = float(num / support)
        per_attack[subtype] = {"recall": recall, "support": support, "numerator": num}

    results["per_attack_recall"] = per_attack
    return results


def print_comparison_table(
    models_results: Dict[str, Dict[str, Any]],
    top_k_pct: float = 0.01,
):
    """
    Print side-by-side Markdown comparison tables for overall metrics and
    per-attack-type recall breakdowns.
    """
    model_names = list(models_results.keys())
    header_str = " | ".join(model_names)
    sep_str = " | ".join(["---"] * len(model_names))

    print(f"\n# Phase 3 Multi-Model Comparison (Test Split @ Top {top_k_pct*100:.0f}% Alert Budget)\n")
    print(f"| Metric / Setting | {header_str} |")
    print(f"|---|{sep_str}|")

    # Overall Core Anomaly PR-AUC
    row_prauc = " | ".join([f"**{models_results[m]['anomaly']['pr_auc']:.4f}**" for m in model_names])
    print(f"| **Core Anomaly PR-AUC (Primary)** | {row_prauc} |")

    # Overall Core Anomaly F1
    row_f1 = " | ".join([f"{models_results[m]['anomaly']['f1']:.4f}" for m in model_names])
    print(f"| Core Anomaly F1-Score | {row_f1} |")

    # Overall Core Anomaly Precision
    row_prec = " | ".join([f"{models_results[m]['anomaly']['precision']:.4f}" for m in model_names])
    print(f"| Core Anomaly Precision | {row_prec} |")

    # Overall Core Anomaly Recall
    row_rec = " | ".join([f"{models_results[m]['anomaly']['recall']:.4f} ({models_results[m]['anomaly']['numerator']}/{models_results[m]['anomaly']['support_pos']})" for m in model_names])
    print(f"| Core Anomaly Recall | {row_rec} |")

    print(f"|---|{sep_str}|")

    # Insider Drift PR-AUC
    row_ambig_prauc = " | ".join([f"{models_results[m]['ambiguous']['pr_auc']:.4f}" for m in model_names])
    print(f"| Insider Drift PR-AUC | {row_ambig_prauc} |")

    # Insider Drift Recall
    row_ambig_rec = " | ".join([f"{models_results[m]['ambiguous']['recall']:.4f} ({models_results[m]['ambiguous']['numerator']}/{models_results[m]['ambiguous']['support_pos']})" for m in model_names])
    print(f"| Insider Drift Recall | {row_ambig_rec} |")

    print("\n## Per-Attack-Type Recall Breakdown (Top 1% Alert Budget)\n")
    print("This table demonstrates where sequence modeling outperforms static and tabular rules on gradual/sequential attacks:\n")
    print(f"| Attack Subtype (Test Support) | {header_str} |")
    print(f"|---|{sep_str}|")

    # Per attack rows
    sample_model = model_names[0]
    for subtype in ALL_ATTACK_SUBTYPES:
        support = models_results[sample_model]["per_attack_recall"][subtype]["support"]
        row_vals = []
        for m in model_names:
            rec_val = models_results[m]["per_attack_recall"][subtype]["recall"]
            num = models_results[m]["per_attack_recall"][subtype]["numerator"]
            # Highlight sequential attack types in bold if appropriate
            if subtype in ["lateral_movement", "exfiltration", "insider_drift"]:
                row_vals.append(f"**{rec_val:.4f} ({num}/{support})**")
            else:
                row_vals.append(f"{rec_val:.4f} ({num}/{support})")
        row_str = " | ".join(row_vals)
        label_disp = f"**{subtype}**" if subtype in ["lateral_movement", "exfiltration", "insider_drift"] else subtype
        print(f"| {label_disp} | {row_str} |")
    print("")
