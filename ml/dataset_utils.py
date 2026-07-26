"""
dataset_utils.py — Feature extraction and causal sequence windowing for Phase 3.

Transforms raw event streams into standardized numeric feature vectors (including
step-based rolling enrichment features) and builds causal sliding windows for
LSTM Autoencoder sequence modeling.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from ml.baseline_profiler import BaselineProfiler


DATA_HEAVY_RESOURCES = {
    "db_customers",
    "s3_data_lake",
    "s3_backups",
    "db_financials",
    "share_exec",
    "share_hr",
}

AUTH_METHODS = ["password", "api_key", "certificate", "mfa", "biometric"]


def extract_event_features(
    df: pd.DataFrame,
    profiler: BaselineProfiler,
    scaler_params: Dict[str, Tuple[float, float]] = None,
    fit_scaler: bool = False,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Tuple[float, float]]]:
    """
    Extract continuous, categorical mismatch, one-hot, and rolling features.
    Standardizes continuous features to zero mean and unit variance.
    """
    out_df = df.copy()
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])

    # Ensure chronological order per entity for sequence/rolling calculations
    out_df = out_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # 1. Time delta log (minutes since previous event for same entity)
    ts_diff = out_df.groupby("entity_id")["timestamp"].diff().dt.total_seconds() / 60.0
    ts_diff = ts_diff.fillna(0.0).clip(lower=0.0)
    out_df["time_delta_log"] = np.log1p(ts_diff)

    # 2. Session duration log
    dur = out_df["session_duration"].fillna(0.0).clip(lower=0.0)
    out_df["session_duration_log"] = np.log1p(dur)

    # 3. Circular hour of day (sin/cos)
    ts = out_df["timestamp"].dt
    h = ts.hour + ts.minute / 60.0
    theta = 2.0 * np.pi * h / 24.0
    out_df["hour_sin"] = np.sin(theta)
    out_df["hour_cos"] = np.cos(theta)

    # 4. Profiler mismatch flags (new geo, new device, new resource)
    new_geos = []
    new_devs = []
    new_res = []
    for (eid, etype), group in out_df.groupby(["entity_id", "entity_type"]):
        prof = profiler.get_profile(str(eid), str(etype))
        
        geos = group["geo_location"].astype(str).to_numpy()
        new_geos.extend(~np.isin(geos, list(prof.known_geos)))
        
        devs = group["device_fingerprint"].astype(str).to_numpy()
        new_devs.extend(~np.isin(devs, list(prof.known_devices)))
        
        res_arr = group["resource_accessed"].astype(str).to_numpy()
        new_res.extend(~np.isin(res_arr, list(prof.known_resources)))

    # Note: because groupby preserves order within group but might reorder across groups if not sorted,
    # let's map back cleanly via index
    mismatch_df = pd.DataFrame(index=out_df.index)
    
    geo_series = pd.Series(0.0, index=out_df.index)
    dev_series = pd.Series(0.0, index=out_df.index)
    res_series = pd.Series(0.0, index=out_df.index)
    heavy_series = pd.Series(0.0, index=out_df.index)

    for (eid, etype), group_idx in out_df.groupby(["entity_id", "entity_type"]).groups.items():
        prof = profiler.get_profile(str(eid), str(etype))
        sub = out_df.loc[group_idx]
        
        geos = sub["geo_location"].astype(str).to_numpy()
        geo_series.loc[group_idx] = (~np.isin(geos, list(prof.known_geos))).astype(float)
        
        devs = sub["device_fingerprint"].astype(str).to_numpy()
        dev_series.loc[group_idx] = (~np.isin(devs, list(prof.known_devices))).astype(float)
        
        res_arr = sub["resource_accessed"].astype(str).to_numpy()
        res_series.loc[group_idx] = (~np.isin(res_arr, list(prof.known_resources))).astype(float)
        heavy_series.loc[group_idx] = np.isin(res_arr, list(DATA_HEAVY_RESOURCES)).astype(float)

    out_df["is_new_geo"] = geo_series
    out_df["is_new_device"] = dev_series
    out_df["is_new_resource"] = res_series
    out_df["is_data_heavy"] = heavy_series

    # 5. One-hot auth methods
    auth_cols = []
    for m in AUTH_METHODS:
        col = f"auth_{m}"
        out_df[col] = (out_df["auth_method"].astype(str) == m).astype(float)
        auth_cols.append(col)

    # 6. Step-based rolling features (10-step causal window per entity)
    # This enriches tabular models (Isolation Forest) with immediate sequence history
    out_df["rolling_time_delta_std_10"] = (
        out_df.groupby("entity_id")["time_delta_log"]
        .transform(lambda s: s.rolling(10, min_periods=1).std().fillna(0.0))
    )
    out_df["rolling_duration_mean_10"] = (
        out_df.groupby("entity_id")["session_duration_log"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )
    out_df["rolling_new_res_count_10"] = (
        out_df.groupby("entity_id")["is_new_resource"]
        .transform(lambda s: s.rolling(10, min_periods=1).sum())
    )
    out_df["rolling_data_heavy_count_10"] = (
        out_df.groupby("entity_id")["is_data_heavy"]
        .transform(lambda s: s.rolling(10, min_periods=1).sum())
    )

    # Define final feature column list
    feature_cols = [
        "time_delta_log",
        "session_duration_log",
        "hour_sin",
        "hour_cos",
        "is_new_geo",
        "is_new_device",
        "is_new_resource",
        "is_data_heavy",
        "rolling_time_delta_std_10",
        "rolling_duration_mean_10",
        "rolling_new_res_count_10",
        "rolling_data_heavy_count_10",
    ] + auth_cols

    # 7. Standardization (zero mean, unit variance)
    if scaler_params is None:
        scaler_params = {}

    if fit_scaler:
        train_mask = out_df["split"] == "train" if "split" in out_df.columns else pd.Series(True, index=out_df.index)
        train_sub = out_df.loc[train_mask, feature_cols]
        for col in feature_cols:
            mean = float(train_sub[col].mean())
            std = float(train_sub[col].std())
            if std < 1e-5:
                std = 1.0
            scaler_params[col] = (mean, std)

    for col in feature_cols:
        mean, std = scaler_params.get(col, (0.0, 1.0))
        out_df[col] = (out_df[col] - mean) / std

    return out_df, feature_cols, scaler_params


def create_sequence_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 10,
) -> np.ndarray:
    """
    Construct causal sliding windows X of shape (N, window_size, D).
    For row i, window i is df.iloc[i - window_size + 1 : i + 1].
    If an entity has fewer preceding rows than window_size, pads by repeating
    the entity's first event.
    
    Assumes df is sorted chronologically by (entity_id, timestamp).
    Returns 3D numpy array matching df rows 1-to-1.
    """
    windows_list = []
    
    # Iterate over entity groups preserving df order
    for eid, group in df.groupby("entity_id", sort=False):
        F = group[feature_cols].to_numpy(dtype=np.float32)
        M, D = F.shape
        if M == 0:
            continue
        
        # Pad by repeating first row (window_size - 1) times
        pad_rows = np.repeat(F[:1], window_size - 1, axis=0)
        F_pad = np.vstack([pad_rows, F])
        
        # Extract causal window ending at each step i from 0 to M-1
        # Window i in F_pad is index [i : i + window_size]
        group_windows = [F_pad[i : i + window_size] for i in range(M)]
        windows_list.extend(group_windows)
        
    all_windows = np.stack(windows_list, axis=0)
    return all_windows
