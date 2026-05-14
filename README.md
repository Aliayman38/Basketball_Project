🏀 AI Basketball Analytics Platform

An AI-powered basketball analytics platform for player tracking, team classification, ball possession analysis, shot detection, court understanding, and advanced game visualization using modern computer vision pipelines.

The system processes basketball footage and produces real-time analytics overlays, top-down court visualizations, possession tracking, speed and distance statistics, shot detection events, and interactive video outputs through a web-based dashboard.

Built using YOLO, RT-DETR, CLIP, BotSORT, OpenCV, PyTorch, and Flask.

Features
Player, referee, ball, and hoop detection
Multi-object tracking with BotSORT
Stable custom player ID management
AI-based team classification using CLIP
Temporal voting and assignment locking for robust team recognition
Ball trajectory interpolation and smoothing
Ball possession analytics
Shot detection system
Court landmark detection
Homography-based top-down court transformation
Speed and distance analytics
Heatmaps and trajectory visualization
Real-time overlay rendering
Interactive Flask web dashboard
Video upload and processing pipeline
CSV and JSON analytics export
Demo Video

Upload the demo video to either:

GitHub Releases
YouTube

Then replace the link below:

## Demo Video

[Watch Demo](https://youtu.be/MdjVjUXRh4M)

[![Watch Demo](https://img.youtube.com/vi/MdjVjUXRh4M/maxresdefault.jpg)](https://youtu.be/MdjVjUXRh4M)

Tech Stack
AI / Computer Vision
PyTorch
OpenCV
YOLO
RT-DETR
CLIP
BotSORT
NumPy
Pandas
Backend
Flask
Analytics
Possession Analysis
Shot Detection
Speed Estimation
Distance Tracking
Court Geometry & Homography
Project Capabilities
🏀 Tracking System

The platform tracks:

Players
Referees
Basketball
Hoop / Net

with stable IDs and trajectory recording.

🏀 Team Classification

Teams are automatically classified using:

CLIP embeddings
Jersey appearance analysis
Temporal majority voting
Multi-signal clustering

without requiring custom jersey training.

🏀 Ball Possession Analytics

The system computes:

Team possession percentages
Player possession percentages
Possession transitions
Per-frame ball ownership

and exports analytics in JSON and CSV formats.

🏀 Shot Detection

The platform detects successful made shots using:

Rim crossing logic
Ball descent validation
Hoop ROI analysis
Trajectory interpolation
🏀 Court Understanding

The system supports:

Court landmark detection
Homography estimation
Top-down court visualization
Spatial player mapping
🏀 Analytics

Generated analytics include:

Player speed
Distance covered
Possession reports
Shot reports
Tracking trajectories
Top-down movement visualization
Installation

Clone the repository:

git clone https://github.com/Aliayman38/Basketball_Project.git
cd Basketball_Project

Install dependencies:

pip install -r requirements.txt
Run the Web Application

Start the Flask dashboard:

python basketball_webapp/app.py

After running the server, open your browser and navigate to:

http://127.0.0.1:5000
Input Videos

Place input basketball videos inside:

input_video/

Supported outputs include:

Tracking videos
Possession overlays
Analytics dashboards
Top-down visualizations
CSV reports
JSON reports
Output Files

Generated outputs are saved inside:

basketball_webapp/static/processed/

Analytics reports are saved inside:

runs/

Including:

distance_report.csv
speed_report.csv
possession_report.json
shots.json
trajectories.json
Project Structure
Basketball_Project/
│
├── basketball_webapp/
├── detection/
├── tracking/
├── team_clustering/
├── src/
│   ├── analytics/
│   └── visualization/
├── models/
├── config/
├── input_video/
├── runs/
└── main.py
Sample Outputs

The platform generates:

Annotated tracking videos
Possession-highlighted videos
Top-down court mapping
Shot event visualizations
Analytics overlays
Player movement reports
