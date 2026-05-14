# 🏀 Basketball Analytics AI Platform

An AI-powered system Using YOLOv26n and RT-DETR-L that processes basketball game footage and produces tracked video, player statistics, shot detection, possession analysis, court mapping, and a web dashboard — fully automated.

---

## 🔍 What It Does

- Detects players, ball, referees, and hoop every frame
- Tracks all objects with stable persistent IDs
- Classifies players into two teams automatically
- Detects made shots and tracks ball possession
- Computes per-player speed and distance
- Maps the court to a top-down bird's-eye view
- Exports everything as video + CSV/JSON reports
- Serves results through a Flask web dashboard

---

## Demo Video

[Watch Demo](https://youtu.be/MdjVjUXRh4M)

[![Watch Demo](https://img.youtube.com/vi/MdjVjUXRh4M/maxresdefault.jpg)](https://youtu.be/MdjVjUXRh4M) 


## ⚙️ Tech Used

| Component | What We Used |
|---|---|
| Object Detection | YOLOv26n + RT-DETR-L — fine-tuned on custom basketball dataset |
| Tracking | BoT-SORT + OC-SORT with OSNet ReID |
| Team Classification |`fusion_core.py` combines K-Means++ color signals with CLIP embeddings using confidence-weighted voting to produce the final team assignment|
| Ball Interpolation | Pandas linear interpolation + rolling average smoothing |
| Shot Detection | Rim-crossing geometry + ball descent validation |
| Court Mapping | Homography (OpenCV) + keypoint detection model |
| Web Dashboard | Flask |

---

## 📁 Project Structure

```
Basketball_Project/
├── main.py                        # Run the full pipeline from CLI
├── basketball_webapp/app.py       # Run the web dashboard
├── detection/                     # YOLOv26n / RT-DETR detection wrappers
├── tracking/                      # BoT-SORT tracker + ball interpolator
├── team_clustering/               # K-Means++ + CLIP team assignment +`fusion_core.py`
├── src/analytics/                 # Shot detection, possession, speed, distance
├── src/analytics/court_detection/ # Homography + top-down view
├── src/visualization/             # Overlay and drawing utilities
├── models/weights/                # Model weight files (.pt)
├── config/                        # YAML configs + homography matrix
└── runs/                          # All outputs land here
```

---

## 🚀 Installation

```bash
# 1. Clone
git clone https://github.com/Aliayman38/Basketball_Project.git
cd Basketball_Project

# 2. Pull model weights (Git LFS)
git lfs install && git lfs pull

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 4. Install PyTorch (pick your CUDA version from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1 — Command Line

```bash
python main.py
```

Edit the variables at the top of `main.py` before running:

```python
VIDEO_PATH   = "input_video/shooting3.mp4"
MODEL_PATH   = "models/weights/last.pt"
DEVICE       = "cuda:0"          # or "cpu"
TEAM_0_DESC  = "yellow jersey"
TEAM_1_DESC  = "dark blue jersey"
```

Outputs are saved to `runs/`.

### Option 2 — Web Dashboard

```bash
python basketball_webapp/app.py
```

Open `http://localhost:5000` → upload a video → click **Process** → view results.

---

## 📊 Outputs

| File | Description |
|---|---|
| `tracking_output.mp4` | Bounding boxes with player IDs and team colors |
| `tracking_possession.mp4` | Possession highlight video |
| `tracking_landmarks.mp4` | Court keypoints overlay |
| `topdown_video.mp4` | Bird's-eye court view |
| `final_output.mp4` | Full dashboard with score and stats |
| `distance_report.csv` | Distance covered per player (meters) |
| `speed_report.csv` | Average and peak speed per player (km/h) |
| `shots.json` | Made shots with timestamps and confidence |
| `possession_report.csv/json` | Possession % per player and team |

---

## 📦 Requirements

- Python 3.9 – 3.11
- NVIDIA GPU with CUDA (recommended)
- Git LFS
