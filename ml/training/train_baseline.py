"""
ORACLE ML — training/train_baseline.py
Runs the baseline learning phase for a machine.

Usage:
    python -m training.train_baseline --asset-id pump_01 --data-dir data/raw/normal/pump_01
"""

import argparse
import os
import json
import numpy as np
import joblib

from ..preprocessing.cleaning import clean_telemetry_payload
from ..preprocessing.normalization import ChannelNormalizer
from ..preprocessing.windowing import (
    window_all_channels,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
)
from ..baseline.baseline_learner import BaselineLearner
from ..baseline.fingerprint import MachineFingerprint
from ..anomaly.isolation_forest import OracleIsolationForest
from ..features.feature_pipeline import build_feature_matrix


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_baseline(asset_id: str, data_dir: str, sample_rate: float = 100.0) -> None:
    """
    Load all normal telemetry files for an asset, learn the baseline,
    train the Isolation Forest, and save all artifacts.
    """
    print(f"\n[ORACLE] Training baseline for asset: {asset_id}")
    print(f"[ORACLE] Loading data from: {data_dir}\n")

    # --- Load all JSON telemetry files ---
    payloads = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r") as f:
            payloads.append(json.load(f))

    if not payloads:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    print(f"[ORACLE] Loaded {len(payloads)} telemetry payloads.")

    # --- Clean + window all payloads ---
    learner = BaselineLearner(asset_id=asset_id, sample_rate=sample_rate)
    normalizer = ChannelNormalizer()
    all_feature_dicts = []

    for i, payload in enumerate(payloads):
        channels = clean_telemetry_payload(payload)
        if i == 0:
            # Fit normalizer on first payload
            channels, normalizer = normalizer.fit_transform(channels), normalizer
        else:
            channels, _ = normalizer.transform(channels), normalizer

        windows = window_all_channels(
            channels,
            window_size=DEFAULT_WINDOW_SIZE,
            step_size=DEFAULT_STEP_SIZE,
        )
        learner.add_windows(windows)

    print(f"[ORACLE] Total windows collected: {learner.window_count()}")

    if not learner.is_ready():
        print(
            f"[WARNING] Only {learner.window_count()} windows collected. "
            f"Minimum recommended: {BaselineLearner.MIN_WINDOWS}. "
            "Proceeding anyway."
        )

    # --- Fit baseline profile ---
    profile = learner.fit()
    print(f"[ORACLE] Baseline fitted. Features: {len(profile.feature_names)}")

    # --- Train Isolation Forest ---
    from features.feature_pipeline import features_to_vector, extract_features_from_windows
    feature_dicts = learner._feature_dicts
    X, feature_names = build_feature_matrix(feature_dicts)
    print(f"[ORACLE] Training Isolation Forest on matrix {X.shape} ...")

    detector = OracleIsolationForest(n_estimators=100, contamination=0.05)
    detector.fit(X)
    print("[ORACLE] Isolation Forest trained.")

    # --- Build fingerprint ---
    fingerprint = MachineFingerprint.from_baseline_profile(profile)
    print(f"[ORACLE] Fingerprint: {fingerprint.to_dict()}")

    # --- Save all artifacts ---
    asset_dir = os.path.join(MODEL_DIR, "trained", asset_id)
    os.makedirs(asset_dir, exist_ok=True)

    profile.save(os.path.join(asset_dir, "baseline_profile.pkl"))
    detector.save(os.path.join(asset_dir, "isolation_forest.pkl"))
    normalizer.save(os.path.join(asset_dir, "normalizer.pkl"))
    joblib.dump(fingerprint, os.path.join(asset_dir, "fingerprint.pkl"))

    # Save metadata
    metadata = {
        "asset_id": asset_id,
        "n_windows": learner.window_count(),
        "n_features": len(profile.feature_names),
        "feature_names": profile.feature_names,
        "fingerprint": fingerprint.to_dict(),
    }
    meta_dir = os.path.join(MODEL_DIR, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{asset_id}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[ORACLE] [OK] All artifacts saved to: {asset_dir}")
    print(f"[ORACLE] [OK] Baseline training complete for {asset_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ORACLE baseline for a machine.")
    parser.add_argument("--asset-id", required=True, help="Machine asset ID")
    parser.add_argument("--data-dir", required=True, help="Directory with normal JSON telemetry")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sensor sample rate (Hz)")
    args = parser.parse_args()

    train_baseline(args.asset_id, args.data_dir, args.sample_rate)
