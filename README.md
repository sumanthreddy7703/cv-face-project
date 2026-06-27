Real-Time Face Detection and Multi-Face Tracking
Authors: Sumanth Reddy Gutha, Babakamal Doddapaneni
This project implements a complete real-time face detection and multi-face tracking pipeline. It compares classical and deep learning detectors (Haar Cascade vs SSD) and correlation-filter trackers (KCF vs CSRT), and includes an IOU-based multi-face tracker with visual analytics.
The entire system runs in real time on CPU using Python and OpenCV.


1. Project Folder Structure

cv-face-project/
  data/
    videos/
      train_video.mp4
      test_video.mp4
  models/
    deploy.prototxt
    haarcascade_frontalface_default.xml
    res10_300x300_ssd_iter_140000.caffemodel
    tracker_config.pkl        # created after training 
  src/
    face_project.py
    detector_compare.py
    tracker_compare.py
  results/                    # will be generated automatically
  requirements.txt
  README.md

  Download Required Videos
To keep the submission size small, the training and testing videos are not included in the zipped project folder.
Both videos can be downloaded from the following Google Drive link:

Google Drive Download Link:
Test Video:
https://drive.google.com/file/d/1hzS5Oo8IQle5Inh6odkiLahi29HpQnN0/view?usp=sharing

Train Video:
https://drive.google.com/file/d/1FhdQNUaTy_OVt-0vTBgWnUBtKCr9FkIW/view?usp=sharing

After downloading, place the videos into:

cv-face-project/data/videos/


2. Installing Dependencies
Create a virtual environment:
macOS / Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt


3. Download Required Models (if missing)
These files MUST be inside the models/ folder:
 SSD DNN Detector
If missing, download from OpenCV:
deploy.prototxt


res10_300x300_ssd_iter_140000.caffemodel


Source:
 https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
 Haar Cascade
If missing, download:
haarcascade_frontalface_default.xml


Source:
 https://github.com/opencv/opencv/tree/master/data/haarcascades


4. Running the Project
Run the main menu (training, testing, webcam)
python src/face_project.py

You will see:
========== MAIN MENU ==========
1 - Real-time webcam detection & tracking
2 - Train tracker hyperparameters on train_video.mp4
3 - Test tracker on test_video.mp4 (with visualizations)
4 - Exit

Option 1 - Real-time webcam detection
Opens webcam


Runs SSD + IOU multi-face tracker


Press q to quit


Option 2 - Train tracker hyperparameters
Uses data/videos/train_video.mp4


Computes track statistics


Automatically selects:


IOU threshold


min_track_length


max_misses


Saves config to models/tracker_config.pkl


Option 3 - Test tracker with visualizations
Generates:
tracking_output.mp4


num_faces_over_time.png


track_length_hist.png


face_heatmap.png


5. Detector Comparison (Haar vs SSD)
Run:
python src/detector_compare.py

Outputs:
FPS


Avg faces per frame


results/detector_fps_compare.png


6. Tracker Comparison (KCF vs CSRT)
Run:
python src/tracker_compare.py

Outputs:
FPS of each tracker


Average frame-to-frame motion


results/tracker_compare.png


If KCF is not available:
pip uninstall -y opencv-python
pip install opencv-contrib-python


7. Expected Outputs
After running all scripts, your results/ folder will contain:
Visual plots used in the report

Comparison charts

Tracking video with IDs