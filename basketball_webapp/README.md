# 🏀 Basketball Analysis Web Application

A complete Flask web dashboard for your basketball computer vision pipeline with an **orange and dark gray (#3D3D3D)** theme.

## 📁 Project Structure

```
basketball_webapp/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Main UI (single-page application)
├── static/
│   ├── css/
│   │   └── style.css       # Orange & gray theme styling
│   ├── js/
│   │   └── app.js          # Frontend interactivity
│   ├── uploads/            # Uploaded videos/images
│   └── processed/          # Pipeline outputs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install flask opencv-python torch numpy
```

### 2. Run the App

```bash
cd basketball_webapp
python app.py
```

Open **http://localhost:5000** in your browser.

## 🎨 Features

### Screens
1. **Upload** — Drag & drop video/image upload with processing pipeline monitor
2. **Detection & Tracking** — Video player with tracking overlays, ball possession stats, shot detection
3. **Homography** — Split view: original camera perspective + top-down court view with player positions
4. **Landmarks** — Court keypoint detection with accuracy metrics and detection status
5. **Dashboard** — Analytics: possession donut chart, speed over time graph, shot log, distance report, final output video

### Score Counter
- Real-time score tracking for both teams (Yellow vs Blue)
- Plus/minus controls with animated feedback
- Reset button

### Theme
- **Primary**: `#FF8C00` (Dark Orange)
- **Background**: `#3D3D3D` (Dark Gray)
- Fully responsive with sidebar navigation

## 🔗 Integrating Your Pipeline

The app is designed to work with your existing `main.py` pipeline. To connect:

### Option A: Direct Import (Recommended)

In `app.py`, replace the mock `run_pipeline()` function with your actual pipeline:

```python
# In app.py, modify the run_pipeline() function:

def run_pipeline():
    """Run your actual basketball analysis pipeline."""
    try:
        # Import your modules
        from tracking.tracker import BasketballTracker
        from src.analytics.dashboard import render_video
        from src.analytics.distance import build_report, export_csv as export_distance_csv
        from src.analytics.speed import build_speed_report, export_csv as export_speed_csv
        from src.analytics.shot_detector import detect_shots, load_trajectory
        from src.analytics.possession import build_possession_report, export_possession_json
        from src.analytics.court_detection.landmarks_overlay import run_landmarks
        from src.analytics.court_detection.homography import run_homography
        from src.analytics.court_detection.topdown_view import run_topdown

        video_path = processing_state['video_path']
        output_dir = Path(app.config['PROCESSED_FOLDER'])

        # 1. Detection & Tracking
        update_progress("Detection & Tracking", 10)
        tracker = BasketballTracker(
            model_path='models/weights/last.pt',
            reid_path='osnet_x0_25_msmt17.pt',
            device=torch.device('cuda:0'),
            team_0_desc="a basketball player wearing a yellow jersey",
            team_1_desc="a basketball player wearing a dark blue jersey",
        )
        # ... (your tracking loop)

        # 2. Trajectory & Analytics
        update_progress("Trajectory Analysis", 30)
        trajectories = tracker.get_trajectories()

        # 3. Shot Detection
        update_progress("Shot Detection", 50)
        shots = detect_shots(...)

        # 4. Landmarks
        update_progress("Court Landmarks", 70)
        run_landmarks(...)

        # 5. Homography
        update_progress("Homography", 85)
        run_homography(...)

        # 6. Dashboard
        update_progress("Dashboard", 95)
        render_video(...)

        # Store results
        with state_lock:
            processing_state['results'] = {
                'tracking_video': str(output_dir / 'tracking.mp4'),
                'possession_video': str(output_dir / 'possession.mp4'),
                'landmarks_video': str(output_dir / 'landmarks.mp4'),
                'topdown_video': str(output_dir / 'topdown.mp4'),
                'final_video': str(output_dir / 'final.mp4'),
                'stats': {
                    'total_frames': tracker.frame_count,
                    'fps': fps,
                    'players_detected': len(trajectories.get('players', {})),
                    'shots_made': len(shots),
                    'possession_team0': f"{report['team_0_pct']:.0f}%",
                    'possession_team1': f"{report['team_1_pct']:.0f}%",
                    'avg_speed': f"{avg_speed:.1f} m/s",
                    'total_distance': f"{total_distance:.1f} km"
                }
            }
            processing_state['progress'] = 100
            processing_state['current_step'] = 'Complete'

    except Exception as e:
        log_message(f"Error: {str(e)}")
    finally:
        with state_lock:
            processing_state['is_processing'] = False
```

### Option B: Subprocess Call

If you prefer to keep your pipeline separate:

```python
import subprocess

def run_pipeline():
    video_path = processing_state['video_path']
    result = subprocess.run(
        ['python', '../main.py', '--video', video_path],
        capture_output=True,
        text=True
    )
    # Parse outputs and update state
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/upload` | POST | Upload video/image |
| `/process` | POST | Start processing pipeline |
| `/status` | GET | Get processing status & logs |
| `/score` | GET/POST | Get/update score |
| `/results` | GET | Get all processing results |
| `/video/<type>` | GET | Stream processed video |
| `/analytics/<type>` | GET | Get analytics data (JSON/CSV) |
| `/reset` | POST | Reset all state |

## 🛠 Customization

### Change Team Colors
Edit CSS variables in `static/css/style.css`:
```css
:root {
    --team-0: #FF8C00;  /* Yellow team */
    --team-1: #1a3a5c;  /* Blue team */
}
```

### Add More Analytics
Extend the dashboard grid in `templates/index.html` and add corresponding chart logic in `static/js/app.js`.

### Model Configuration
Update the model paths in `app.py` or pass them via environment variables:
```bash
export MODEL_PATH="models/weights/last.pt"
export REID_PATH="osnet_x0_25_msmt17.pt"
```

## 📱 Mobile Support

The app is fully responsive:
- **Desktop**: Full sidebar + wide video panels
- **Tablet**: Collapsed sidebar icons + stacked layouts
- **Mobile**: Hidden sidebar (tap to open) + single column

## 🔒 Notes

- Max upload size: **500MB**
- Supports: MP4, AVI, MOV, MKV, JPG, PNG, WEBP
- Processing runs in a background thread (non-blocking)
- All state is server-side (refresh-safe during processing)

---

Built for your basketball CV pipeline 🏀
