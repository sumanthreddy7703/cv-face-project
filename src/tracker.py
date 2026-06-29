"""IOU-based multi-face tracker.

A lightweight multi-object tracker that associates detections to existing
tracks by greedy Intersection-over-Union matching. It assigns each face a
persistent ID, tolerates short detection gaps (``max_misses``), and keeps a
record of every track ever created (``all_tracks``) for offline statistics.

This module is intentionally free of OpenCV / drawing dependencies so the
tracking logic can be imported and unit-tested on its own.
"""

from collections import deque
from typing import List, Sequence, Tuple

# A box is (x, y, w, h); a detection additionally carries a confidence.
Box = Tuple[int, int, int, int]


def compute_iou(boxA: Box, boxB: Box) -> float:
    """Intersection over Union between two boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    union = float(boxAArea + boxBArea - interArea + 1e-8)
    return interArea / union


class Track:
    """Represents a single tracked face over time."""

    def __init__(self, track_id: int, bbox: Box, frame_idx: int):
        self.id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.last_frame = frame_idx
        self.misses = 0
        self.history = deque(maxlen=100)
        self.history.append(self.center())

    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (int(x + w / 2), int(y + h / 2))


class SimpleIOUTracker:
    """Simple multi-object tracker based on IOU + greedy matching.

    We keep *all* tracks ever created in ``all_tracks`` (for statistics).
    """

    def __init__(self, max_misses: int = 5, iou_threshold: float = 0.3):
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.all_tracks: List[Track] = []   # store every track ever created
        self.next_id = 1

    def update(self, detections: Sequence[Sequence], frame_idx: int) -> List[Track]:
        """Update tracker with detections from the current frame.

        Args:
            detections: list of (x, y, w, h, conf)
            frame_idx: int

        Returns:
            list of Track objects (active tracks)
        """
        det_boxes = [d[:4] for d in detections]
        assigned_dets = set()

        # Match existing tracks with detections (greedy IOU)
        for track in self.tracks:
            best_iou = 0.0
            best_j = -1
            for j, det_box in enumerate(det_boxes):
                if j in assigned_dets:
                    continue
                iou = compute_iou(track.bbox, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != -1 and best_iou >= self.iou_threshold:
                # matched
                track.bbox = det_boxes[best_j]
                track.last_frame = frame_idx
                track.misses = 0
                track.history.append(track.center())
                assigned_dets.add(best_j)
            else:
                # missed in this frame
                track.misses += 1

        # Create new tracks for unmatched detections
        for j, det_box in enumerate(det_boxes):
            if j not in assigned_dets:
                new_track = Track(self.next_id, det_box, frame_idx)
                self.tracks.append(new_track)
                self.all_tracks.append(new_track)
                self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return self.tracks
