"""Visualization helpers for drawing tracked faces onto frames."""

from typing import List

import numpy as np
import cv2

from tracker import Track


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    """Draw bounding boxes, IDs, and short trajectories for each track."""
    for track in tracks:
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track.id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw trajectory
        for i in range(1, len(track.history)):
            cv2.line(frame, track.history[i - 1], track.history[i], (0, 255, 0), 1)

    return frame
