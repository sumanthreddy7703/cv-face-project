"""Interactive entry point for the real-time face detection & tracking project.

This is a thin menu that wires together the modular components:
    - detector.FaceDetectorDNN   (SSD face detector)
    - tracker.SimpleIOUTracker   (IOU multi-face tracker, used by the modes)
    - realtime.run_realtime      (live webcam mode)

Additional modes (training, testing + visual analytics) are added in later
steps of the project.
"""

from config import PROTOTXT_PATH, MODEL_PATH
from detector import FaceDetectorDNN
from config_io import load_tracker_config
from realtime import run_realtime


def main():
    detector = FaceDetectorDNN(PROTOTXT_PATH, MODEL_PATH)

    while True:
        print("\n========== MAIN MENU ==========")
        print("1 - Real-time webcam detection & tracking")
        print("2 - Exit")
        choice = input("Enter choice (1/2): ").strip()

        if choice == "1":
            config = load_tracker_config()
            run_realtime(detector, config)

        elif choice == "2":
            print("[INFO] Exiting.")
            break

        else:
            print("[WARN] Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
