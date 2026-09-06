"""
ORACLE ML — preprocessing/cleaning.py
Cleans raw sensor readings received from the ESP32 / backend.
"""

import numpy as np
from typing import Dict, List, Any


# Sensor channels that ORACLE processes
SENSOR_CHANNELS = ["ax", "ay", "az", "vibration", "acoustic", "temperature"]


def remove_none_values(readings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop any reading dict that is None or missing required keys.
    """
    required = {"ax", "ay", "az", "vibration"}
    cleaned = []
    for r in readings:
        if r is None:
            continue
        if not required.issubset(r.keys()):
            continue
        cleaned.append(r)
    return cleaned


def clip_outliers(values: np.ndarray, std_factor: float = 5.0) -> np.ndarray:
    """
    Clip values that are more than `std_factor` standard deviations from the mean.
    This removes sensor glitches without distorting the baseline.
    """
    if len(values) == 0:
        return values
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values
    lower = mean - std_factor * std
    upper = mean + std_factor * std
    return np.clip(values, lower, upper)


def fill_missing_channels(readings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For optional channels (acoustic, temperature), fill None with forward-fill
    or the channel mean so downstream code always gets a float.
    """
    optional_channels = ["acoustic", "temperature"]
    # collect last known good value
    last_known: Dict[str, float] = {}

    result = []
    for r in readings:
        r_copy = dict(r)
        for ch in optional_channels:
            val = r_copy.get(ch)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # use last known, or 0 if we have nothing yet
                r_copy[ch] = last_known.get(ch, 0.0)
            else:
                last_known[ch] = float(val)
        result.append(r_copy)
    return result


def extract_channel(readings: List[Dict[str, Any]], channel: str) -> np.ndarray:
    """
    Pull a single sensor channel out of the reading list and return as ndarray.
    """
    values = [float(r.get(channel, 0.0)) for r in readings]
    return np.array(values, dtype=np.float64)


def clean_telemetry_payload(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Main entry point for cleaning a telemetry payload from the backend.

    Expected payload format:
        {
            "asset_id": "machine_001",
            "timestamp": 1234567890,
            "readings": [
                {"t": 0.0, "ax": 0.1, "ay": 0.2, "az": 9.8,
                 "vibration": 0.05, "acoustic": 120.0, "temperature": 35.0},
                ...
            ]
        }

    Returns a dict of channel -> cleaned np.ndarray.
    """
    readings = payload.get("readings", [])

    if not readings:
        raise ValueError(f"Empty readings in payload for asset {payload.get('asset_id')}")

    # Step 1: remove broken rows
    readings = remove_none_values(readings)

    if not readings:
        raise ValueError("All readings were invalid after null-removal")

    # Step 2: fill optional channels
    readings = fill_missing_channels(readings)

    # Step 3: extract & clip each channel
    cleaned: Dict[str, np.ndarray] = {}
    for ch in SENSOR_CHANNELS:
        raw = extract_channel(readings, ch)
        cleaned[ch] = clip_outliers(raw)

    return cleaned
