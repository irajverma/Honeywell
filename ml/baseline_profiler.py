"""
baseline_profiler.py — Phase 2: Per-Entity Statistical Profiling Model.

Learns normal behavioral profiles from the training split of events.csv,
implements a configurable cold-start fallback to population-level statistics,
scores test events using weighted deviation/z-score logic, and evaluates PR-AUC
separately for core anomalies vs. ambiguous insider drift.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@dataclass
class EntityBaseline:
    """Encapsulates the statistical baseline profile for an entity or population."""
    entity_id: str
    entity_type: str
    event_count: int
    hour_hist: np.ndarray  # 24-bin normalized histogram (with Laplace smoothing)
    hour_circ_mean: float  # Circular mean of login hour [0, 24)
    hour_circ_std: float   # Circular standard deviation of login hour (clamped >= 0.5)
    known_geos: Set[str] = field(default_factory=set)
    known_resources: Set[str] = field(default_factory=set)
    duration_mean: float = 0.0
    duration_std: float = 1.0
    known_devices: Set[str] = field(default_factory=set)
    is_population_fallback: bool = False


def _compute_circular_stats(hours: np.ndarray) -> Tuple[float, float]:
    """
    Compute circular mean and standard deviation for hours of the day (0..24).
    Handles midnight wrapping smoothly.
    """
    if len(hours) == 0:
        return 12.0, 6.0

    # Convert hours to radians [0, 2pi)
    theta = 2.0 * np.pi * hours / 24.0
    sin_mean = np.mean(np.sin(theta))
    cos_mean = np.mean(np.cos(theta))

    # Circular mean
    mean_rad = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)
    circ_mean = mean_rad * 24.0 / (2.0 * np.pi)

    # Circular distance from mean for each point
    diffs = np.abs(hours - circ_mean)
    circ_diffs = np.minimum(diffs, 24.0 - diffs)
    circ_std = float(np.sqrt(np.mean(circ_diffs ** 2)))
    
    # Clamp std to avoid division by zero or overly brittle z-scores
    circ_std = max(0.5, circ_std)
    return float(circ_mean), circ_std


def _compute_profile_from_df(
    df: pd.DataFrame,
    entity_id: str,
    entity_type: str,
    is_fallback: bool = False
) -> EntityBaseline:
    """Compute an EntityBaseline from a dataframe of training events."""
    n_events = len(df)
    if n_events == 0:
        # Return empty default profile
        return EntityBaseline(
            entity_id=entity_id,
            entity_type=entity_type,
            event_count=0,
            hour_hist=np.ones(24) / 24.0,
            hour_circ_mean=12.0,
            hour_circ_std=6.0,
            known_geos=set(),
            known_resources=set(),
            duration_mean=5.0,
            duration_std=2.0,
            known_devices=set(),
            is_population_fallback=is_fallback,
        )

    # Ensure timestamp is datetime
    ts = pd.to_datetime(df["timestamp"])
    hours = ts.dt.hour + ts.dt.minute / 60.0
    hours_arr = hours.to_numpy()

    # 1. Hour histogram (integer 0..23) with Laplace smoothing
    int_hours = ts.dt.hour.to_numpy()
    counts, _ = np.histogram(int_hours, bins=np.arange(25))
    smoothed_counts = counts + 1.0  # Laplace smoothing
    hour_hist = smoothed_counts / smoothed_counts.sum()

    # 2. Circular hour stats
    circ_mean, circ_std = _compute_circular_stats(hours_arr)

    # 3. Sets for categorical features
    known_geos = set(df["geo_location"].dropna().astype(str).unique())
    known_resources = set(df["resource_accessed"].dropna().astype(str).unique())
    known_devices = set(df["device_fingerprint"].dropna().astype(str).unique())

    # 4. Session duration mean and std
    durations = df["session_duration"].dropna().to_numpy()
    if len(durations) > 0:
        dur_mean = float(np.mean(durations))
        dur_std = max(0.5, float(np.std(durations)))
    else:
        dur_mean, dur_std = 5.0, 2.0

    return EntityBaseline(
        entity_id=entity_id,
        entity_type=entity_type,
        event_count=n_events,
        hour_hist=hour_hist,
        hour_circ_mean=circ_mean,
        hour_circ_std=circ_std,
        known_geos=known_geos,
        known_resources=known_resources,
        duration_mean=dur_mean,
        duration_std=dur_std,
        known_devices=known_devices,
        is_population_fallback=is_fallback,
    )


class BaselineProfiler:
    """
    Per-entity statistical profiler with population-level cold-start fallback.
    """

    def __init__(
        self,
        min_events: int = 10,
        weights: Dict[str, float] = None,
        z_thresholds: Dict[str, float] = None,
    ):
        self.min_events = min_events
        self.weights = weights or {
            "new_geo": 3.0,
            "unknown_device": 3.0,
            "new_resource": 2.0,
            "off_hours": 1.5,
            "duration_outlier": 1.0,
        }
        self.z_thresholds = z_thresholds or {
            "hour_z": 2.0,
            "duration_z": 2.5,
        }
        self.profiles: Dict[str, EntityBaseline] = {}
        self.pop_profiles: Dict[str, EntityBaseline] = {}

    def fit(self, events_df: pd.DataFrame):
        """Fit baseline profiles on the training split of events_df."""
        df = events_df.copy()
        if "split" in df.columns:
            df = df[df["split"] == "train"]

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. Fit population-level fallback profiles for each entity_type
        self.pop_profiles.clear()
        for etype, group in df.groupby("entity_type"):
            pop_profile = _compute_profile_from_df(
                group,
                entity_id=f"POPULATION_{etype.upper()}",
                entity_type=str(etype),
                is_fallback=True,
            )
            self.pop_profiles[str(etype)] = pop_profile
        print(f"  [Profiler] Fitted {len(self.pop_profiles)} population fallback profiles "
              f"({', '.join(self.pop_profiles.keys())})")

        # 2. Fit individual entity profiles
        self.profiles.clear()
        n_individual = 0
        n_fallback = 0

        for eid, group in df.groupby("entity_id"):
            etype = str(group["entity_type"].iloc[0]) if not group.empty else "user"
            if len(group) >= self.min_events:
                profile = _compute_profile_from_df(group, entity_id=str(eid), entity_type=etype, is_fallback=False)
                self.profiles[str(eid)] = profile
                n_individual += 1
            else:
                # Will fall back to population profile during scoring
                n_fallback += 1

        print(f"  [Profiler] Fitted {n_individual} individual profiles "
              f"(>={self.min_events} events). {n_fallback} entities will use population fallback.")

    def get_profile(self, entity_id: str, entity_type: str) -> EntityBaseline:
        """Retrieve individual profile if event_count >= min_events, else fallback."""
        if entity_id in self.profiles:
            return self.profiles[entity_id]
        # Fallback to population profile
        pop_profile = self.pop_profiles.get(entity_type)
        if pop_profile is not None:
            return pop_profile
        # Extreme fallback if entity_type never seen
        return EntityBaseline(
            entity_id=entity_id,
            entity_type=entity_type,
            event_count=0,
            hour_hist=np.ones(24) / 24.0,
            hour_circ_mean=12.0,
            hour_circ_std=6.0,
            known_geos=set(),
            known_resources=set(),
            duration_mean=5.0,
            duration_std=2.0,
            known_devices=set(),
            is_population_fallback=True,
        )

    def score_event(self, row: pd.Series, profile: EntityBaseline = None) -> Tuple[float, str]:
        """Score a single event series against its baseline profile."""
        if profile is None:
            profile = self.get_profile(str(row["entity_id"]), str(row.get("entity_type", "user")))

        score = 0.0
        reasons = []

        # 1. Geo location check
        geo = str(row.get("geo_location", ""))
        if geo and geo not in profile.known_geos:
            score += self.weights["new_geo"]
            reasons.append("new_geo_location")

        # 2. Device fingerprint check
        dev = str(row.get("device_fingerprint", ""))
        if dev and dev not in profile.known_devices:
            score += self.weights["unknown_device"]
            reasons.append("unknown_device_fingerprint")

        # 3. Resource accessed check
        res = str(row.get("resource_accessed", ""))
        if res and res not in profile.known_resources:
            score += self.weights["new_resource"]
            reasons.append("new_resource_accessed")

        # 4. Off-hours check
        ts = pd.to_datetime(row["timestamp"])
        h = ts.hour + ts.minute / 60.0
        diff = min(abs(h - profile.hour_circ_mean), 24.0 - abs(h - profile.hour_circ_mean))
        z_hour = diff / profile.hour_circ_std
        if z_hour > self.z_thresholds["hour_z"]:
            score += self.weights["off_hours"]
            reasons.append("off_hours_access")

        # 5. Session duration outlier check
        dur = float(row.get("session_duration", 0.0))
        z_dur = abs(dur - profile.duration_mean) / profile.duration_std
        if z_dur > self.z_thresholds["duration_z"]:
            score += self.weights["duration_outlier"]
            reasons.append("session_duration_outlier")

        reason_str = ";".join(reasons) if reasons else "none"
        return score, reason_str

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized/grouped scoring for an entire dataframe."""
        out_df = df.copy()
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])
        
        scores_list = np.zeros(len(out_df), dtype=float)
        reasons_list = ["none"] * len(out_df)

        # Group by (entity_id, entity_type) for fast vectorized evaluation
        for (eid, etype), group in out_df.groupby(["entity_id", "entity_type"]):
            profile = self.get_profile(str(eid), str(etype))
            indices = group.index.to_numpy()

            group_scores = np.zeros(len(group), dtype=float)
            group_reasons = [[] for _ in range(len(group))]

            # 1. Geo
            geos = group["geo_location"].astype(str).to_numpy()
            geo_mask = ~np.isin(geos, list(profile.known_geos))
            group_scores[geo_mask] += self.weights["new_geo"]
            for idx in np.where(geo_mask)[0]:
                group_reasons[idx].append("new_geo_location")

            # 2. Device
            devs = group["device_fingerprint"].astype(str).to_numpy()
            dev_mask = ~np.isin(devs, list(profile.known_devices))
            group_scores[dev_mask] += self.weights["unknown_device"]
            for idx in np.where(dev_mask)[0]:
                group_reasons[idx].append("unknown_device_fingerprint")

            # 3. Resource
            res = group["resource_accessed"].astype(str).to_numpy()
            res_mask = ~np.isin(res, list(profile.known_resources))
            group_scores[res_mask] += self.weights["new_resource"]
            for idx in np.where(res_mask)[0]:
                group_reasons[idx].append("new_resource_accessed")

            # 4. Hour
            ts = group["timestamp"].dt
            h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
            diffs = np.abs(h - profile.hour_circ_mean)
            circ_diffs = np.minimum(diffs, 24.0 - diffs)
            z_hours = circ_diffs / profile.hour_circ_std
            hour_mask = z_hours > self.z_thresholds["hour_z"]
            group_scores[hour_mask] += self.weights["off_hours"]
            for idx in np.where(hour_mask)[0]:
                group_reasons[idx].append("off_hours_access")

            # 5. Duration
            durs = group["session_duration"].to_numpy(dtype=float)
            z_durs = np.abs(durs - profile.duration_mean) / profile.duration_std
            dur_mask = z_durs > self.z_thresholds["duration_z"]
            group_scores[dur_mask] += self.weights["duration_outlier"]
            for idx in np.where(dur_mask)[0]:
                group_reasons[idx].append("session_duration_outlier")

            scores_list[indices] = group_scores
            for i, idx in enumerate(indices):
                if group_reasons[i]:
                    reasons_list[idx] = ";".join(group_reasons[i])

        out_df["baseline_score"] = scores_list
        out_df["flagged_reasons"] = reasons_list
        return out_df

    def evaluate(self, scored_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate precision, recall, F1, and PR-AUC against ground truth.
        Reports metrics separately for 'anomaly' vs 'ambiguous' labels.
        """
        # Ensure timestamp matching format
        df_left = scored_df.copy()
        df_right = gt_df[["entity_id", "timestamp", "label", "attack_subtype"]].copy()
        df_left["timestamp"] = df_left["timestamp"].astype(str)
        df_right["timestamp"] = df_right["timestamp"].astype(str)

        merged = df_left.merge(
            df_right,
            on=["entity_id", "timestamp"],
            how="inner"
        )
        if merged.empty:
            raise ValueError("Merged dataframe is empty! Check timestamp formats or split matching.")

        results = {}

        # ── 1. Core Anomaly Evaluation (normal vs anomaly) ───────────
        anomaly_mask = merged["label"].isin(["normal", "anomaly"])
        eval_anomaly = merged[anomaly_mask].copy()
        y_true_anom = (eval_anomaly["label"] == "anomaly").astype(int)
        y_score_anom = eval_anomaly["baseline_score"].to_numpy()

        pr_auc_anom = float(average_precision_score(y_true_anom, y_score_anom))
        
        # Metrics at Threshold >= 2.0
        y_pred_anom_2 = (y_score_anom >= 2.0).astype(int)
        prec_anom_2, rec_anom_2, f1_anom_2, _ = precision_recall_fscore_support(
            y_true_anom, y_pred_anom_2, average="binary", zero_division=0
        )

        # Metrics at Top 1% Alert Budget
        k_anom = max(1, int(len(y_score_anom) * 0.01))
        thresh_1pct = np.sort(y_score_anom)[-k_anom]
        y_pred_anom_1pct = (y_score_anom >= thresh_1pct).astype(int)
        prec_anom_1pct, rec_anom_1pct, f1_anom_1pct, _ = precision_recall_fscore_support(
            y_true_anom, y_pred_anom_1pct, average="binary", zero_division=0
        )

        results["anomaly"] = {
            "pr_auc": pr_auc_anom,
            "prec_thresh_2.0": float(prec_anom_2),
            "rec_thresh_2.0": float(rec_anom_2),
            "f1_thresh_2.0": float(f1_anom_2),
            "prec_top_1pct": float(prec_anom_1pct),
            "rec_top_1pct": float(rec_anom_1pct),
            "f1_top_1pct": float(f1_anom_1pct),
            "support_pos": int(y_true_anom.sum()),
            "support_neg": int((y_true_anom == 0).sum()),
        }

        # ── 2. Ambiguous / Insider Drift Evaluation (normal vs ambiguous) ───
        ambig_mask = merged["label"].isin(["normal", "ambiguous"])
        eval_ambig = merged[ambig_mask].copy()
        y_true_ambig = (eval_ambig["label"] == "ambiguous").astype(int)
        y_score_ambig = eval_ambig["baseline_score"].to_numpy()

        pr_auc_ambig = float(average_precision_score(y_true_ambig, y_score_ambig))
        
        y_pred_ambig_2 = (y_score_ambig >= 2.0).astype(int)
        prec_ambig_2, rec_ambig_2, f1_ambig_2, _ = precision_recall_fscore_support(
            y_true_ambig, y_pred_ambig_2, average="binary", zero_division=0
        )

        k_ambig = max(1, int(len(y_score_ambig) * 0.01))
        thresh_ambig_1pct = np.sort(y_score_ambig)[-k_ambig]
        y_pred_ambig_1pct = (y_score_ambig >= thresh_ambig_1pct).astype(int)
        prec_ambig_1pct, rec_ambig_1pct, f1_ambig_1pct, _ = precision_recall_fscore_support(
            y_true_ambig, y_pred_ambig_1pct, average="binary", zero_division=0
        )

        results["ambiguous"] = {
            "pr_auc": pr_auc_ambig,
            "prec_thresh_2.0": float(prec_ambig_2),
            "rec_thresh_2.0": float(rec_ambig_2),
            "f1_thresh_2.0": float(f1_ambig_2),
            "prec_top_1pct": float(prec_ambig_1pct),
            "rec_top_1pct": float(rec_ambig_1pct),
            "f1_top_1pct": float(f1_ambig_1pct),
            "support_pos": int(y_true_ambig.sum()),
            "support_neg": int((y_true_ambig == 0).sum()),
        }

        return results
