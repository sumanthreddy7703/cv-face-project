"""Training mode: tune tracker hyperparameters from train_video.mp4.

This runs detection + IOU tracking over the training video, gathers statistics
about how long tracks survive, and picks a sensible ``min_track_length`` from
the 25th percentile of observed track lengths. The resulting config is pickled
to TRACKER_CONFIG_PATH for the testing / webcam modes to load.
"""

import os
import pickle

import cv2
import numpy as np

from config import VIDEOS_DIR, TRACKER_CONFIG_PATH, DEFAULT_TRACKER_CONFIG
from utils import ensure_file, ensure_dir
from tracker import SimpleIOUTracker


def train_on_video(detector):
    """Estimate good tracker hyperparameters from the training video and save them."""
    video_path = os.path.join(VIDEOS_DIR, "train_video.mp4")
    ensure_file(video_path, "Training video (train_video.mp4)")

    print(f"[INFO] Training on video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open training video.")
        return

    tracker = SimpleIOUTracker(
        max_misses=DEFAULT_TRACKER_CONFIG["max_misses"],
        iou_threshold=DEFAULT_TRACKER_CONFIG["iou_threshold"],
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracker.update(detections, frame_idx)
        frame_idx += 1

    cap.release()

    all_tracks = tracker.all_tracks
    track_lengths = [len(t.history) for t in all_tracks if len(t.history) > 0]
    if not track_lengths:
        print("[WARN] No tracks found during training. Try a different video.")
        return

    track_lengths = np.array(track_lengths)
    avg_len = float(np.mean(track_lengths))
    med_len = float(np.median(track_lengths))
    p25 = float(np.percentile(track_lengths, 25))

    print("\n[INFO] Training statistics:")
    print(f"  Number of tracks: {len(track_lengths)}")
    print(f"  Average track length: {avg_len:.2f} frames")
    print(f"  Median track length: {med_len:.2f} frames")
    print(f"  25th percentile track length: {p25:.2f} frames")

    min_track_length = max(3, int(round(p25)))
    max_misses = DEFAULT_TRACKER_CONFIG["max_misses"]

    config = {
        "max_misses": max_misses,
        "min_track_length": min_track_length,
        "iou_threshold": DEFAULT_TRACKER_CONFIG["iou_threshold"],
    }

    ensure_dir(os.path.dirname(TRACKER_CONFIG_PATH))
    with open(TRACKER_CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

    print("\n[INFO] Trained tracker configuration:")
    print(config)
    print(f"[INFO] Saved tracker config to: {TRACKER_CONFIG_PATH}")
