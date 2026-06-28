"""SSD (Res10) deep-learning face detector built on OpenCV's DNN module.

The detector wraps OpenCV's Caffe-based Res10 300x300 SSD model and returns
filtered face boxes. Filtering removes detections that are too small or have an
implausible aspect ratio, which keeps the downstream tracker clean.
"""

from typing import List, Tuple

import cv2
import numpy as np

from config import (
    CONF_THRESHOLD,
    MIN_FACE_SIZE,
    ASPECT_RATIO_MIN,
    ASPECT_RATIO_MAX,
)
from utils import ensure_file

# A detection is (x, y, w, h, confidence).
Detection = Tuple[int, int, int, int, float]


class FaceDetectorDNN:
    """Face detector based on OpenCV's Res10 SSD model."""

    def __init__(self, prototxt_path, model_path, conf_threshold=CONF_THRESHOLD):
        ensure_file(prototxt_path, "Prototxt")
        ensure_file(model_path, "Caffe model")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a BGR frame.

        Returns:
            list of (x, y, w, h, conf)
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes: List[Detection] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            bw = x2 - x1
            bh = y2 - y1
            if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                continue

            aspect = bw / float(bh + 1e-8)
            if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
                continue

            boxes.append((x1, y1, bw, bh, confidence))

        return boxes
