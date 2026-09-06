"""
ORACLE ML — training/evaluate_model.py
Evaluates the trained anomaly detection model using:
- Normal data (expected: low anomaly score)
- Faulty data (expected: high anomaly score)

Usage:
    python -m training.evaluate_model --asset-id pump_01
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
from ..anomaly.anomaly_detector import AnomalyDetector
from ..baseline.baseline_profile import BaselineProfile

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_payloads(directory: str):
    payloads = []
    if not os.path.isdir(directory):
        return payloads
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                payloads.append(json.load(f))
    return payloads


def run_evaluation(asset_id: str) -> None:
    print(f"\n[ORACLE] Evaluating model for: {asset_id}\n")

    asset_dir = os.path.join(MODEL_DIR, "trained", asset_id)
    profile_path = os.path.join(asset_dir, "baseline_profile.pkl")
    model_path   = os.path.join(asset_dir, "isolation_forest.pkl")
    norm_path    = os.path.join(asset_dir, "normalizer.pkl")

    # Load artifacts
    profile    = BaselineProfile.load(profile_path)
    normalizer = ChannelNormalizer.load(norm_path)
    detector   = AnomalyDetector(asset_id=asset_id)
    detector.load(model_path, profile)

    results = {}

    for label, subdir in [("NORMAL", "normal"), ("FAULTY", "faulty")]:
        data_dir = os.path.join(DATA_DIR, "raw", subdir)
        payloads = load_payloads(data_dir)

        if not payloads:
            print(f"[SKIP] No {label} data found in {data_dir}")
            continue

        scores = []
        for payload in payloads:
            channels = clean_telemetry_payload(payload)
            channels = normalizer.transform(channels)
            windows = window_all_channels(channels, DEFAULT_WINDOW_SIZE, DEFAULT_STEP_SIZE)
            result = detector.detect(windows)
            scores.append(result["anomaly_score"])

        results[label] = scores
        mean_score = np.mean(scores)
        print(f"[{label}] Payloads: {len(scores)} | Mean Anomaly Score: {mean_score:.4f}")

        if label == "NORMAL" and mean_score > 0.4:
            print(f"  [WARNING] Normal data scoring high -- check baseline quality.")
        if label == "FAULTY" and mean_score < 0.5:
            print(f"  [WARNING] Faulty data not detected -- consider retraining.")

    if "NORMAL" in results and "FAULTY" in results:
        sep = np.mean(results["FAULTY"]) - np.mean(results["NORMAL"])
        print(f"\n[ORACLE] Separation (faulty_mean - normal_mean): {sep:.4f}")
        if sep > 0.2:
            print("[ORACLE] [OK] Model shows good separation between normal and faulty.")
        else:
            print("[ORACLE] [WARNING] Separation is low. Consider collecting more training data.")

    print("\n[ORACLE] Evaluation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    args = parser.parse_args()
    run_evaluation(args.asset_id)
