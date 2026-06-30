"""Interactive entry point for the real-time face detection & tracking project.

This is a thin menu that wires together the modular components:
    - detector.FaceDetectorDNN   (SSD face detector)
    - tracker.SimpleIOUTracker   (IOU multi-face tracker, used by the modes)
    - realtime.run_realtime      (live webcam mode)
    - train.train_on_video       (tune tracker hyperparameters)

The testing + visual analytics mode is added in a later step of the project.
"""

from config import PROTOTXT_PATH, MODEL_PATH
from detector import FaceDetectorDNN
from config_io import load_tracker_config
from realtime import run_realtime
from train import train_on_video


def main():
    detector = FaceDetectorDNN(PROTOTXT_PATH, MODEL_PATH)

    while True:
        print("\n========== MAIN MENU ==========")
        print("1 - Real-time webcam detection & tracking")
        print("2 - Train tracker hyperparameters on train_video.mp4")
        print("3 - Exit")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            config = load_tracker_config()
            run_realtime(detector, config)

        elif choice == "2":
            train_on_video(detector)

        elif choice == "3":
            print("[INFO] Exiting.")
            break

        else:
            print("[WARN] Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
