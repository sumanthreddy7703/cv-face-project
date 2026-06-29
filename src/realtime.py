"""Real-time webcam mode: live face detection + IOU multi-face tracking."""

import cv2

from tracker import SimpleIOUTracker
from drawing import draw_tracks


def run_realtime(detector, config):
    """Real-time webcam detection + IOU-based tracking."""
    tracker = SimpleIOUTracker(
        max_misses=config["max_misses"],
        iou_threshold=config["iou_threshold"],
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam started. Press 'q' to quit.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        vis = draw_tracks(frame.copy(), tracks)

        cv2.putText(vis, "Real-time face detection & tracking - press 'q' to quit",
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Real-Time Face Detection & Tracking", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
