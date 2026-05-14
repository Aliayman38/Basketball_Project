🏀 Basketball Analytics AI Platform
An end-to-end AI-powered basketball analytics platform that processes raw game footage and produces tracked video, tactical overlays, player statistics, shot detection, court mapping, and a web dashboard — fully automated.


## Demo Video

[Watch Demo](https://youtu.be/MdjVjUXRh4M)

[![Watch Demo](https://img.youtube.com/vi/MdjVjUXRh4M/maxresdefault.jpg)](https://youtu.be/MdjVjUXRh4M) 

📋 Table of Contents

Overview
Full Pipeline
Project Structure
Module Details

Object Detection
Multi-Object Tracking
Team Classification
Ball Tracking & Interpolation
Possession Analytics
Shot Detection
Court Understanding
Speed & Distance Analytics
Visualization Engine
Flask Web Dashboard


Models & Weights
Generated Outputs
Requirements
Installation
How to Run


🔍 Overview
This platform takes a basketball video file and runs it through a full AI pipeline that:

Detects players, the ball, referees, and the hoop every frame
Tracks all objects with stable persistent IDs using BoT-SORT + a custom ID manager
Classifies players into two teams using CLIP embeddings and jersey color analysis — no manual labeling
Interpolates missing ball detections and smooths its trajectory
Computes frame-level possession ownership with debounce logic
Detects made shots using rim-crossing and ball descent validation
Maps the court from camera view to a top-down bird's-eye perspective using homography
Computes per-player speed, distance, and possession stats and exports them as CSV/JSON
Streams everything through a Flask web dashboard for upload, processing, and playback


⚙️ Full Pipeline
Video Input
     │
     ▼
Object Detection  ──────────────────────  YOLO / RT-DETR
     │                                    players · ball · referees · hoop
     ▼
Multi-Object Tracking  ─────────────────  BoT-SORT + OC-SORT + OSNet ReID
     │                                    Kalman stabilization
     ▼
Custom Stable ID Assignment  ────────────  Cooldown recycling · grace frames
     │                                    occlusion handling
     ▼
Team Classification  ────────────────────  CLIP embeddings · jersey analysis
     │                                    temporal voting · assignment locking
     ▼
Ball Tracking & Interpolation  ──────────  linear interpolation · gap filling
     │                                    rolling average smoothing
     ▼
Analytics Engine
     ├── Possession analytics  ──────────  per-frame · per-player · per-team
     ├── Shot detection  ─────────────────  rim crossing · descent validation
     ├── Speed & distance  ───────────────  FPS-aware · meters-per-pixel
     └── Court mapping  ──────────────────  homography · top-down view
     │
     ▼
Visualization & Overlay Rendering  ──────  bounding boxes · labels · heatmaps
     │                                    team colors · possession highlights
     ▼
Web Dashboard Output  ───────────────────  Flask · upload · process · playback

📁 Project Structure
Basketball_Project/
│
├── main.py                              # CLI entry point — runs full pipeline
│
├── basketball_webapp/
│   ├── app.py                           # Flask web application entry point
│   ├── templates/                       # HTML templates
│   └── static/                          # CSS and JavaScript
│
├── detection/                           # Detection wrappers and utilities
│
├── tracking/
│   ├── tracker.py                       # Main BasketballTracker class (BoT-SORT)
│   ├── ocsort_tracker.py                # OC-SORT tracker alternative
│   └── interpolator.py                  # Ball trajectory interpolation
│
├── team_clustering/
│   ├── clusterer.py                     # CLIP-based team assignment
│   └── fusion_core.py                   # Multi-signal fusion and temporal voting
│
├── src/
│   ├── tracker.py                       # Core tracking utilities
│   ├── analytics/
│   │   ├── possession.py                # Frame-level possession computation
│   │   ├── possession_overlay.py        # Possession highlight video renderer
│   │   ├── shot_detector.py             # Made-shot detection engine
│   │   ├── speed.py                     # Per-player speed computation
│   │   ├── distance.py                  # Per-player distance computation
│   │   └── court_detection/
│   │       ├── homography.py            # Homography estimation
│   │       ├── topdown_view.py          # Bird's-eye court projection
│   │       └── landmarks_overlay.py     # Court keypoint overlay renderer
│   └── visualization/
│       └── visualizer.py               # General drawing and overlay utilities
│
├── overlay.py                           # Composite overlay renderer
│
├── models/
│   └── weights/
│       ├── last.pt                      # Custom YOLOv26n detection weights
│       ├── court_kp.pt                  # Court landmark keypoint weights
│       └── rtdetr-l.pt                  # RT-DETR weights
│
├── config/                              # YAML configuration files
├── input_video/                         # Input videos go here
└── runs/                                # All pipeline outputs land here

🧩 Module Details
1. Object Detection
Files: detection/
Two models are supported and used together:

YOLO — primary real-time detector, fine-tuned on a custom basketball dataset
RT-DETR — transformer-based detector used as an alternative/fallback

Detected classes per frame:

player
ball
referee
hoop / net

Model weights are stored in models/weights/.

2. Multi-Object Tracking
Files: tracking/tracker.py, tracking/ocsort_tracker.py, src/tracker.py
The tracking system uses BoT-SORT and OC-SORT from the boxmot library with several custom improvements on top:
FeatureImplementationReID modelOSNet-x0.25 (osnet_x0_25_msmt17.pt) — maintains identity through occlusionsKalman filterNumerically stabilized to prevent ID drift on slow-moving playersID recyclingCooldown-based — IDs are only reassigned after a configurable cooldown framesGrace framesLost tracks are kept alive for N frames before being droppedStable assignmentCustom ID manager ensures no ID swaps between players
Referee tracking is handled separately from player tracking.

3. Team Classification
Files: team_clustering/clusterer.py, team_clustering/fusion_core.py
This is one of the most sophisticated parts of the system. It uses CLIP (zero-shot vision-language model) instead of a trained jersey classifier:
Pipeline:

Crop the jersey region from the player bounding box
Run CLIP similarity against two text descriptions (e.g. "yellow jersey", "dark blue jersey")
Apply a confidence threshold to the similarity scores
Use temporal voting — accumulate votes across multiple frames before assigning a team
Lock the assignment once confidence is high — the team label does not change again

fusion_core.py combines CLIP signals with HSV jersey-color clustering for more robust assignment in difficult lighting conditions.

4. Ball Tracking & Interpolation
File: tracking/interpolator.py
The ball is frequently occluded or missed by the detector (motion blur, fast movement). The interpolator handles this:

Linear interpolation fills gaps when the ball is missing for a short window
DataFrame interpolation using Pandas for smooth trajectory reconstruction
Rolling average smoothing reduces noise in detected positions

This produces a clean, continuous ball trajectory that the shot detector and possession module depend on.

5. Possession Analytics
Files: src/analytics/possession.py, src/analytics/possession_overlay.py
Frame-by-frame possession is computed by measuring the Euclidean distance between the ball center and every player bounding box center. The closest player within a threshold owns possession.
Extra logic:

Debounce — a minimum number of consecutive frames is required before a possession change is confirmed (prevents flickering)
Transition tracking — every change of possession is logged with its timestamp

Outputs:

possession_report.csv — possession frames and percentage per player and team
possession_report.json — same in JSON

Visualization from possession_overlay.py:

Glowing bounding box around the player with the ball
"HAS BALL" text label
Team-colored possession bar


6. Shot Detection
File: src/analytics/shot_detector.py
A real shot detection engine — not a simple proximity check:
CheckLogicRim crossingBall center crosses the hoop ROI boundaryDescent validationBall must be moving downward (y increasing) at the crossing pointNet exit confirmationBall must exit below the net regionHoop ROI analysisOnly activates when ball is inside the defined hoop region of interestCooldownAfter a made shot is logged, a cooldown window prevents duplicate detectionsInterpolated trajectoryWorks on the smoothed ball trajectory from the interpolator
Outputs:

Shot timestamps (start frame, end frame)
Confidence scores
shots.json report


7. Court Understanding
Files: src/analytics/court_detection/homography.py, topdown_view.py, landmarks_overlay.py
The court is understood geometrically in three steps:

Landmark detection — a keypoint model (court_kp.pt) detects court line intersections (corners, three-point line endpoints, paint corners, etc.)
Homography estimation — homography.py computes a perspective transform matrix from detected landmarks to a canonical top-down court template
Top-down projection — topdown_view.py applies the homography to player positions and produces a bird's-eye tactical view

This enables real-world spatial analysis: distances, positioning heatmaps, and tactical movement patterns are all computed in actual court coordinates.

8. Speed & Distance Analytics
Files: src/analytics/speed.py, src/analytics/distance.py
Player positions from the tracker are converted to real-world coordinates using the METERS_PER_PIXEL scale factor.
Distance (distance.py):

Sums Euclidean displacement between consecutive frames for each player
Outputs distance_report.csv with total distance in meters

Speed (speed.py):

Computes instantaneous speed per frame using position delta and FPS
Aggregates to average and peak speed in km/h
Outputs speed_report.csv

Both modules support optionally excluding referees via the INCLUDE_REFEREES flag.

9. Visualization Engine
Files: src/visualization/visualizer.py, overlay.py
All visual overlays are rendered by this module onto the output video:

Team-colored bounding boxes for players
Persistent ID labels
Distance and speed labels per player
Ball trajectory trail
"HAS BALL" possession indicator
Shot flash effect on made baskets
Top-down court minimap
Analytics dashboard panel


10. Flask Web Dashboard
File: basketball_webapp/app.py
A full web interface for the platform:

Upload a video through the browser
Trigger the processing pipeline on the server
Play back the processed output video
View analytics reports (possession, speed, distance, shots)

Frontend is plain HTML/CSS/JS inside basketball_webapp/templates/ and basketball_webapp/static/.

🧠 Models & Weights
FilePurposemodels/weights/last.ptCustom YOLO detection model (players, ball, hoop, referee)models/weights/court_kp.ptCourt landmark keypoint detectionmodels/weights/rtdetr-l.ptRT-DETR alternative detectorosnet_x0_25_msmt17.ptOSNet ReID — auto-downloaded by boxmot on first run

Git LFS: All .pt files are tracked via Git LFS. Run git lfs pull after cloning or the weight files will be empty stubs.


📊 Generated Outputs
Videos
FileDescriptiontracking_*.mp4Raw tracking output with bounding boxes and IDstracking_possession.mp4Possession highlight videotracking_landmarks.mp4Court keypoint and line overlaytopdown_*.mp4Bird's-eye court projection videofinal_output.mp4Full dashboard with score, stats panel, and shot flashes
Reports
FileDescriptionanalytics/trajectories.jsonPer-frame position log for every player and the ballanalytics/distance_report.csvTotal distance covered (meters) per playeranalytics/speed_report.csvAverage and peak speed (km/h) per playeranalytics/shots.jsonMade shots: frames, coordinates, confidenceanalytics/possession_report.csvPossession time and percentage per player and teamanalytics/possession_report.jsonSame possession data in JSON

📦 Requirements

Python 3.9 – 3.11
NVIDIA GPU with CUDA (CPU supported but very slow for video processing)
Git LFS


🚀 Installation
bash# 1. Clone
git clone https://github.com/Aliayman38/Basketball_Project.git
cd Basketball_Project

# 2. Pull model weights (Git LFS)
git lfs install
git lfs pull

# 3. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 4. Install PyTorch with CUDA first → https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install all other dependencies
pip install -r requirements.txt

▶️ How to Run
Option 1 — Command line (full pipeline)
bashpython main.py
Edit the configuration variables at the top of main.py before running:
VariableDefaultDescriptionVIDEO_PATHinput_video/game.mp4Path to input videoMODEL_PATHmodels/weights/last.ptYOLO weightsDEVICEcuda:0Torch deviceMETERS_PER_PIXEL0.0264Real-world scale factorTEAM_0_DESC"yellow jersey"CLIP description for Team 0TEAM_1_DESC"dark blue jersey"CLIP description for Team 1BALL_POSSESSION_THRESHOLD80.0Max pixel distance for possessionPOSSESSION_MIN_FRAMES3Frames to confirm possession changeINCLUDE_REFEREESFalseInclude referees in stat reports
All outputs are saved under runs/.

Option 2 — Web dashboard
bashpython basketball_webapp/app.py
Then open your browser at:
http://localhost:5000
Upload a video, click Process, and watch the pipeline run. When done, the dashboard shows the annotated video alongside all analytics reports.

🛠 Tech Stack
ComponentTechnologyObject DetectionUltralytics YOLO + RT-DETRTrackingBoT-SORT + OC-SORT via boxmotRe-IdentificationOSNet-x0.25Team ClassificationCLIP (open-clip-torch) + HSV K-MeansBall InterpolationPandas linear interpolation + rolling averageShot DetectionRim-crossing geometry + descent validationCourt MappingHomography via OpenCVVideo I/OOpenCVAnalytics & ReportsPandas + JSONWeb DashboardFlaskVisualisationOpenCV + MatplotlibDataset ManagementRoboflow
