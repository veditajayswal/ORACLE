"""
ORACLE ML — preprocessing/windowing.py
Converts a continuous stream of sensor readings into fixed-size
overlapping windows that the feature extractor processes one at a time.
"""

import numpy as np
from typing import Dict, List, Generator


# Defaults — can be overridden per machine
DEFAULT_WINDOW_SIZE = 128   # number of samples per window
DEFAULT_STEP_SIZE = 64      # hop between consecutive windows (50% overlap)


def sliding_windows(
    arr: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> Generator[np.ndarray, None, None]:
    """
    Yield successive overlapping windows from a 1-D array.
    Windows that are shorter than window_size at the end are discarded.
    """
    n = len(arr)
    start = 0
    while start + window_size <= n:
        yield arr[start : start + window_size]
        start += step_size


def window_all_channels(
    channels: Dict[str, np.ndarray],
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> List[Dict[str, np.ndarray]]:
    """
    Apply sliding window to every channel simultaneously.
    Each returned element is one time-window across all channels.

    Returns:
        List of dicts, each dict maps channel_name -> window ndarray (length = window_size)
    """
    if not channels:
        return []

    # Use the shortest channel to determine window count
    min_len = min(len(v) for v in channels.values())
    if min_len < window_size:
        # Not enough data — return the single available segment, zero-padded
        padded = {}
        for ch, arr in channels.items():
            pad_len = window_size - len(arr)
            padded[ch] = np.pad(arr[:window_size], (0, pad_len), mode="constant")
        return [padded]

    # Collect all window positions
    positions = []
    start = 0
    while start + window_size <= min_len:
        positions.append(start)
        start += step_size

    windows = []
    for pos in positions:
        window = {}
        for ch, arr in channels.items():
            window[ch] = arr[pos : pos + window_size]
        windows.append(window)

    return windows


def get_window_count(
    n_samples: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> int:
    """Return how many complete windows fit in n_samples."""
    if n_samples < window_size:
        return 1  # we pad
    count = 0
    start = 0
    while start + window_size <= n_samples:
        count += 1
        start += step_size
    return count
