🏀 Basketball Analytics AI Pipeline
🚀 Project Overview
This project implements a modular AI system to analyze basketball gameplay. It transitions from raw video frames to deep statistical insights through a multi-stage computer vision pipeline.

🏗 System Architecture (The Pipeline)
Input: Raw Broadcast/Tactical Video (1080p/4K).

Training & Detection (Ali): YOLO11-Large + K-Means Clustering.

Temporal Association (Omari): Multi-Object Tracking (MOT).

Spatial Mapping (Rashid): Homography & Top-Down Projection.

Analytics (Rababah): Kinematic & Statistical Reporting.

📋 Project Roadmap & Task Distribution
1. Training & Detection & Team Clustering
Owner: Ali Al-Qoraan

Modules: src/detection/, src/team_clustering/, models/weights/

Accomplishments:

Advanced Detection: Developed and fine-tuned a YOLO11-Large model at 1280px resolution, achieving 93.9% mAP.

Unsupervised Team Clustering: Engineered a modular TeamClusterer using K-Means Clustering to automatically segment players into two teams based on dominant jersey color.

Model Optimization: Managed training over 75 epochs to ensure high-precision detection of the ball, players, and hoop.

Deliverable: Enriched data stream (BBoxes + Class Labels + Team IDs).

2. Multi-Object Tracking (MOT)
Owner: Mohammad Al-Omari

Module: src/tracking/

Objective: Integrate the BasketballDetector with ByteTrack or BoT-SORT.

Deliverable: Maintain persistent Unique IDs for players and the ball across frames, ensuring each ID is correctly linked to its assigned team (from Ali's clusterer).

3. Spatial Analytics & Court Mapping
Owner: Rashid

Module: src/analytics/

Objective: Map 2D pixel coordinates to a Bird's Eye View (Top-down) of the basketball court.

Deliverable: Generate tactical Heatmaps and spatial occupancy maps for team formation analysis.

4. Performance Metrics & Stats
Owner: Rababah

Module: src/analytics/

Objective: Extract data-driven metrics including Player Speed, Total Distance Covered, and Shot/Jump Statistics.

Deliverable: A comprehensive performance report generated for every player ID.

🛠 Setup & Contribution
Sync Repository: git pull origin main

Handle Large Files: git lfs pull (Required for .pt weights).

Dependencies: pip install ultralytics scikit-learn opencv-python torch

Execution: python main.py