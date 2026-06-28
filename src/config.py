"""Central paths and configuration constants for the face project.

Keeping these in one place means every module (detector, tracker, app,
comparison scripts) agrees on where the models, videos and results live, and
on the detection/tracking defaults.
"""

import os

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
PROTOTXT_PATH = os.path.join(ROOT_DIR, "models", "deploy.prototxt")
HAAR_PATH = os.path.join(ROOT_DIR, "models", "haarcascade_frontalface_default.xml")

TRACKER_CONFIG_PATH = os.path.join(ROOT_DIR, "models", "tracker_config.pkl")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
VIDEOS_DIR = os.path.join(ROOT_DIR, "data", "videos")

# ----- Detection config (slightly relaxed to work well with FDDB images) -----
CONF_THRESHOLD = 0.6
MIN_FACE_SIZE = 40
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 1.8

# ----- Default tracking hyperparameters -----
DEFAULT_TRACKER_CONFIG = {
    "max_misses": 5,        # max frames a track can be missing before deletion
    "min_track_length": 5,  # min frames to consider a track "valid"
    "iou_threshold": 0.3,   # IOU threshold for matching detections to tracks
}
