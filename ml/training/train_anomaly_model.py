"""
ORACLE ML — training/train_anomaly_model.py
Trains or retrains the Isolation Forest anomaly detection model for a machine.

Usage:
    python -m training.train_anomaly_model --asset-id pump_01 --data-dir ../data/raw/normal
"""

import argparse
import os
import json
import numpy as np

from ..preprocessing.cleaning import clean_telemetry_payload
from ..preprocessing.normalization import ChannelNormalizer
from ..preprocessing.windowing import (
    window_all_channels,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
)
from ..features.feature_pipeline import (
    extract_features_from_windows,
    build_feature_matrix,
    features_to_vector,
    extract_window_features,
)
from ..anomaly.isolation_forest import OracleIsolationForest
from ..baseline.baseline_profile import BaselineProfile

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_anomaly_model(
    asset_id: str,
    data_dir: str,
    n_estimators: int = 100,
    contamination: float = 0.05,
    sample_rate: float = 100.0,
) -> None:
    """
    Loads normal operating data, uses the machine's baseline profile
    and normalizer, trains the Isolation Forest model, and saves it.
    """
    print(f"\n[ORACLE] Training Anomaly Model (Isolation Forest) for: {asset_id}")
    print(f"[ORACLE] Source data directory: {data_dir}")

    asset_dir = os.path.join(MODEL_DIR, "trained", asset_id)
    profile_path = os.path.join(asset_dir, "baseline_profile.pkl")
    norm_path = os.path.join(asset_dir, "normalizer.pkl")

    # 1. Check if baseline profile exists
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"Baseline profile not found at {profile_path}. "
            f"Please run train_baseline.py first to establish machine baseline."
        )

    profile = BaselineProfile.load(profile_path)
    normalizer = ChannelNormalizer.load(norm_path) if os.path.exists(norm_path) else None

    # 2. Load telemetry JSON files
    payloads = []
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Directory not found: {data_dir}")

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname), "r") as f:
                payloads.append(json.load(f))

    if not payloads:
        raise ValueError(f"No JSON telemetry files found in {data_dir}")

    print(f"[ORACLE] Loaded {len(payloads)} normal telemetry payloads.")

    # 3. Clean, normalize, window, and extract features
    feature_vectors = []
    for payload in payloads:
        channels = clean_telemetry_payload(payload)
        if normalizer:
            channels = normalizer.transform(channels)
        windows = window_all_channels(
            channels,
            window_size=DEFAULT_WINDOW_SIZE,
            step_size=DEFAULT_STEP_SIZE,
        )
        for w in windows:
            feat_dict = extract_window_features(w, sample_rate=sample_rate)
            vec = features_to_vector(feat_dict, profile.feature_names)
            feature_vectors.append(vec)

    if len(feature_vectors) < 10:
        raise ValueError(
            f"Isolation Forest requires at least 10 sample windows to train. "
            f"Found only {len(feature_vectors)}. Please provide more data files."
        )

    X = np.vstack(feature_vectors)
    print(f"[ORACLE] Extracted training matrix with shape: {X.shape}")

    # 4. Train Isolation Forest
    print(f"[ORACLE] Fitting Isolation Forest (estimators={n_estimators}, contamination={contamination})...")
    detector = OracleIsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    detector.fit(X)

    # 5. Save model artifact
    out_model_path = os.path.join(asset_dir, "isolation_forest.pkl")
    detector.save(out_model_path)

    # 6. Update metadata
    meta_path = os.path.join(MODEL_DIR, "metadata", f"{asset_id}_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    metadata["isolation_forest"] = {
        "n_estimators": n_estimators,
        "contamination": contamination,
        "training_windows": X.shape[0],
        "n_features": X.shape[1],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ORACLE] [OK] Model saved to: {out_model_path}")
    print(f"[ORACLE] [OK] Anomaly model training complete for {asset_id}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly model.")
    parser.add_argument("--asset-id", required=True, help="Asset identifier (e.g. pump_01)")
    parser.add_argument("--data-dir", default="../data/raw/normal", help="Path to normal JSON files")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees (default: 100)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination rate (default: 0.05)")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sampling rate in Hz (default: 100.0)")
    args = parser.parse_args()

    train_anomaly_model(
        asset_id=args.asset_id,
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        sample_rate=args.sample_rate,
    )
