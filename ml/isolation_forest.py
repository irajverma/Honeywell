"""
isolation_forest.py — Unsupervised tabular machine learning baseline (Phase 3).

Wraps scikit-learn's IsolationForest to learn normal event distributions from
continuous features and rolling step enrichments.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    """
    Unsupervised tabular baseline using scikit-learn's IsolationForest.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = "auto",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """Fit Isolation Forest on normal training split events."""
        train_df = df.copy()
        if "split" in train_df.columns:
            train_df = train_df[train_df["split"] == "train"]

        X = train_df[feature_cols].to_numpy(dtype=np.float32)
        print(f"  [IForest] Fitting IsolationForest on {len(X):,} training rows (dim={len(feature_cols)})...")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        print("  [IForest] Fitting complete.")

    def score(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Score events using -decision_function(X).
        In scikit-learn, decision_function returns negative values for anomalies,
        so negating it ensures higher scores indicate greater anomaly severity.
        """
        if self.model is None:
            raise RuntimeError("IsolationForestModel must be fitted before scoring!")

        X = df[feature_cols].to_numpy(dtype=np.float32)
        raw_scores = self.model.decision_function(X)
        return -raw_scores
