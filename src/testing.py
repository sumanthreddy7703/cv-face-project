"""Testing mode: run the tracker on test_video.mp4 and produce visual analytics.

Outputs written to the results/ folder:
    - tracking_output.mp4     annotated video with per-face IDs and trajectories
    - num_faces_over_time.png number of active tracks per frame
    - track_length_hist.png   histogram of (valid) track lengths
    - face_heatmap.png        where faces appeared most, overlaid on a frame
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import VIDEOS_DIR, RESULTS_DIR
from utils import ensure_file, ensure_dir
from tracker import SimpleIOUTracker
from drawing import draw_tracks


def test_on_video(detector, config):
    """Run detection + tracking on the test video and save video + plots."""
    video_path = os.path.join(VIDEOS_DIR, "test_video.mp4")
    ensure_file(video_path, "Test video (test_video.mp4)")

    print(f"[INFO] Testing on video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open testing video.")
        return

    ensure_dir(RESULTS_DIR)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(RESULTS_DIR, "tracking_output.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker = SimpleIOUTracker(
        max_misses=config["max_misses"],
        iou_threshold=config["iou_threshold"],
    )

    frame_idx = 0
    num_tracks_per_frame = []
    heatmap = np.zeros((height, width), dtype=np.float32)
    sample_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if sample_frame is None:
            sample_frame = frame.copy()

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        # visualization frame
        vis = draw_tracks(frame.copy(), tracks)
        writer.write(vis)

        num_tracks_per_frame.append(len(tracks))

        # accumulate heatmap
        for t in tracks:
            x, y, w, h = t.bbox
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            if x1 > x0 and y1 > y0:
                heatmap[y0:y1, x0:x1] += 1.0

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[INFO] Saved tracking video to: {out_path}")

    # track length stats
    valid_lengths = [len(t.history) for t in tracker.all_tracks
                     if len(t.history) >= config["min_track_length"]]

    # ----- Visualization 1: #faces over time -----
    plt.figure()
    plt.plot(num_tracks_per_frame)
    plt.xlabel("Frame index")
    plt.ylabel("Number of active tracks")
    plt.title("Number of faces tracked over time")
    plot1_path = os.path.join(RESULTS_DIR, "num_faces_over_time.png")
    plt.savefig(plot1_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {plot1_path}")

    # ----- Visualization 2: histogram of track lengths -----
    if valid_lengths:
        plt.figure()
        plt.hist(valid_lengths, bins=20)
        plt.xlabel("Track length (frames)")
        plt.ylabel("Count")
        plt.title("Histogram of track lengths")
        plot2_path = os.path.join(RESULTS_DIR, "track_length_hist.png")
        plt.savefig(plot2_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved plot: {plot2_path}")
    else:
        print("[WARN] No valid tracks for histogram (maybe video too short).")

    # ----- Visualization 3: face presence heatmap -----
    if sample_frame is not None and np.max(heatmap) > 0:
        hm_norm = heatmap / np.max(heatmap)

        plt.figure(figsize=(8, 4.5))
        plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
        plt.imshow(hm_norm, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title("Face presence heatmap (red = faces appeared often)")

        heatmap_path = os.path.join(RESULTS_DIR, "face_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[INFO] Saved face heatmap: {heatmap_path}")
    else:
        print("[WARN] Could not generate heatmap (no faces or no frames).")
