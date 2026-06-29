"""Loading (and reporting) of trained tracker hyperparameters."""

import os
import pickle

from config import TRACKER_CONFIG_PATH, DEFAULT_TRACKER_CONFIG


def load_tracker_config():
    """Load tracker config saved by the training step, or fall back to defaults."""
    if os.path.exists(TRACKER_CONFIG_PATH):
        try:
            with open(TRACKER_CONFIG_PATH, "rb") as f:
                config = pickle.load(f)
            print(f"[INFO] Loaded tracker config from {TRACKER_CONFIG_PATH}")
            return config
        except Exception as e:
            print("[WARN] Failed to load tracker config, using defaults:", e)
    else:
        print("[INFO] No tracker config found, using defaults.")

    return DEFAULT_TRACKER_CONFIG.copy()
