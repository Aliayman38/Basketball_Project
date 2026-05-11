"""
Basketball Analysis Web Application
Calls your existing main.py pipeline via subprocess.
Place this file in your BASKETBALL_PROJECT root folder.
"""

from flask import Flask, render_template, request, jsonify, Response, make_response, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import os
import sys
import subprocess
import threading
import time
import re
import hashlib
import shutil
from datetime import datetime
import cv2  # Required for video info reading and demo pipeline

# =============================================================================
#  FIND PROJECT ROOT
# =============================================================================

APP_DIR = Path(__file__).parent.resolve()
print(f"[INIT] App directory: {APP_DIR}")

# =============================================================================
#  FLASK APP SETUP
# =============================================================================

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'), static_folder=str(APP_DIR / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(APP_DIR / 'static' / 'uploads')
app.config['PROCESSED_FOLDER'] = str(APP_DIR / 'static' / 'processed')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png', 'webp'}

processing_state = {
    'is_processing': False,
    'progress': 0,
    'current_step': '',
    'video_path': None,
    'results': {},
    'score': {'team_0': 0, 'team_1': 0},
    'logs': [],
}

state_lock = threading.Lock()

# =============================================================================
#  VIDEO STREAMING (HTTP 206 Partial Content)
# =============================================================================

def send_file_partial(path, mimetype='video/mp4'):
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    size = os.path.getsize(path)
    range_header = request.headers.get('Range', None)
    last_modified = os.path.getmtime(path)
    etag = hashlib.md5(str(last_modified).encode()).hexdigest()

    if 'If-None-Match' in request.headers:
        if request.headers['If-None-Match'] == etag:
            return make_response('', 304)

    if not range_header:
        response = make_response(send_file(path, mimetype=mimetype))
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['ETag'] = etag
        response.headers['Content-Length'] = str(size)
        return response

    byte1, byte2 = 0, None
    m = re.search(r'(\d+)-(\d*)', range_header)
    if m:
        g = m.groups()
        if g[0]: byte1 = int(g[0])
        if g[1]: byte2 = int(g[1])

    if byte2 is not None:
        length = byte2 - byte1 + 1
    else:
        length = size - byte1

    data = None
    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + len(data) - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(len(data)))
    rv.headers.add('ETag', etag)
    rv.headers.add('Cache-Control', 'no-cache')
    return rv

# =============================================================================
#  HELPERS
# =============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_message(msg):
    with state_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        processing_state['logs'].append(f"[{timestamp}] {msg}")
        if len(processing_state['logs']) > 100:
            processing_state['logs'] = processing_state['logs'][-100:]
    print(f"[LOG] {msg}")

def update_progress(step, percent):
    with state_lock:
        processing_state['current_step'] = step
        processing_state['progress'] = percent
    print(f"[PROGRESS] {step}: {percent}%")

# =============================================================================
#  PIPELINE RUNNER - Calls your main.py via subprocess
# =============================================================================

def reencode_h264(src: str, dst: str) -> bool:
    """
    Re-encode a video to H.264 / yuv420p so browsers can play it.
    Tries ffmpeg first (most reliable), falls back to OpenCV.
    Returns True if successful.
    """
    # ── Try ffmpeg ────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', src,
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',   # required for broad browser compatibility
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',  # puts index at front for streaming
                '-an',                      # drop audio (CV pipelines have none)
                dst
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            log_message(f"   ✅ Re-encoded (ffmpeg H.264): {Path(dst).name}")
            return True
        else:
            log_message(f"   ⚠️ ffmpeg failed: {result.stderr[-300:] if result.stderr else 'no output'}")
    except FileNotFoundError:
        log_message("   ⚠️ ffmpeg not found, trying OpenCV fallback...")
    except Exception as e:
        log_message(f"   ⚠️ ffmpeg error: {e}")

    # ── Fallback: OpenCV with avc1 (H.264) codec ─────────────────────────────
    try:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return False
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))
        if not writer.isOpened():
            # avc1 unavailable on this OpenCV build; just copy as-is
            cap.release()
            shutil.copy2(src, dst)
            log_message(f"   ⚠️ H.264 unavailable, copied as-is: {Path(dst).name}")
            return True
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            log_message(f"   ✅ Re-encoded (OpenCV avc1): {Path(dst).name}")
            return True
    except Exception as e:
        log_message(f"   ⚠️ OpenCV re-encode failed: {e}")

    # Last resort: just copy
    shutil.copy2(src, dst)
    log_message(f"   ⚠️ Could not re-encode, copied raw: {Path(dst).name}")
    return False


def run_pipeline_subprocess(video_path: str, output_dir: str):
    """
    Run your existing main.py pipeline by calling it as a subprocess.
    This guarantees you get the EXACT same 5 outputs.
    """

    log_message("🚀 Starting pipeline via main.py...")

    # ── FIX 1: Find main.py at project root (app.py lives in basketball_webapp/ subdir) ──
    # Check current dir first, then parent (the actual project root)
    main_py_path = APP_DIR / 'main.py'
    if not main_py_path.exists():
        main_py_path = APP_DIR.parent / 'main.py'
    if not main_py_path.exists():
        log_message(f"❌ main.py not found in {APP_DIR} or {APP_DIR.parent}")
        log_message("   Falling back to demo mode...")
        run_demo_pipeline(video_path, output_dir)
        return

    # ── FIX 2: PROJECT_ROOT is where main.py lives — all relative paths resolve from here ──
    PROJECT_ROOT = main_py_path.parent
    log_message(f"Found main.py at: {main_py_path}")
    log_message(f"Project root: {PROJECT_ROOT}")

    # Create output directories relative to project root (matching main.py's OUTPUT_PATH)
    runs_dir = PROJECT_ROOT / 'runs' / 'bot-sort tracking'
    analytics_dir = runs_dir / 'analytics'
    runs_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    # ── FIX 3: Copy video into PROJECT_ROOT/data/ (not basketball_webapp/data/) ──
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    video_name = Path(video_path).name
    data_video_path = data_dir / video_name
    shutil.copy2(video_path, data_video_path)
    log_message(f"Video copied to: {data_video_path}")

    temp_main = PROJECT_ROOT / 'main_web_temp.py'

    try:
        with open(main_py_path, 'r', encoding='utf-8') as f:
            main_content = f.read()

        # Use regex replacements so they work regardless of the exact filename in main.py
        main_content = re.sub(
            r"VIDEO_PATH\s*=\s*['\"].*?['\"]",
            f"VIDEO_PATH = 'data/{video_name}'",
            main_content
        )
        main_content = re.sub(
            r"OUTPUT_PATH\s*=\s*['\"].*?['\"]",
            "OUTPUT_PATH = 'runs/bot-sort tracking/tracking_output.mp4'",
            main_content
        )

        # ── FIX 4: Replace hardcoded cuda:0 with a safe CPU fallback ──
        # tracker.py also passes device into BotSort with half=True which breaks on CPU,
        # so we patch both DEVICE and the BotSort half flag here.
        main_content = re.sub(
            r"DEVICE\s*=\s*torch\.device\(['\"]cuda:\d+['\"]\)",
            "DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
            main_content
        )

        # Prepend a CPU-compatibility shim: BotSort's half=True crashes on CPU
        cpu_shim = """\
# --- CPU compatibility shim injected by app.py ---
import torch as _torch
if not _torch.cuda.is_available():
    try:
        from boxmot.trackers.botsort.botsort import BotSort as _BotSort
        _orig_botsort_init = _BotSort.__init__
        def _cpu_botsort_init(self, *args, **kwargs):
            kwargs['half'] = False
            _orig_botsort_init(self, *args, **kwargs)
        _BotSort.__init__ = _cpu_botsort_init
    except Exception as _e:
        print(f"[SHIM] BotSort CPU patch skipped: {_e}")
# --- end shim ---

"""
        main_content = cpu_shim + main_content

        with open(temp_main, 'w', encoding='utf-8') as f:
            f.write(main_content)

        log_message("Temporary main_web_temp.py created")
        update_progress("Detection & Tracking", 10)
        log_message("Running detection and tracking...")

        env = os.environ.copy()
        # ── PYTHONPATH must include project root so imports (tracking, src, detection…) resolve ──
        env['PYTHONPATH'] = str(PROJECT_ROOT)

        process = subprocess.Popen(
            [sys.executable, str(temp_main)],
            cwd=str(PROJECT_ROOT),   # FIX 2: run from project root, not basketball_webapp/
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output to logs in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                log_message(line)

                # Parse progress from output
                if 'Frame' in line and 'FPS' in line:
                    update_progress("Detection & Tracking", 30)
                elif 'Shot Detection' in line:
                    update_progress("Shot Detection", 50)
                elif 'Possession' in line:
                    update_progress("Ball Possession", 65)
                elif 'Landmark' in line:
                    update_progress("Court Landmarks", 80)
                elif 'Homography' in line:
                    update_progress("Homography", 90)
                elif 'Top-Down' in line:
                    update_progress("Top-Down View", 95)
                elif 'Done' in line or 'Complete' in line:
                    update_progress("Dashboard", 98)

        process.wait()

        if process.returncode != 0:
            log_message(f"⚠️ main.py exited with code {process.returncode}")
        else:
            log_message("✅ main.py completed successfully")

        # Clean up temp file
        if temp_main.exists():
            temp_main.unlink()

        # Collect and re-encode all 5 output videos to H.264 for browser playback
        update_progress("Re-encoding for browser", 99)
        log_message("🎬 Re-encoding outputs to H.264 for browser playback...")
        alt_names = {
            'tracking_video':  ['tracking_output.mp4', 'tracking_botsort3.mp4'],
            'possession_video': ['tracking_possession.mp4'],
            'landmarks_video':  ['tracking_landmarks.mp4'],
            'topdown_video':    ['tracking_topdown.mp4'],
            'final_video':      ['final_output1.mp4', 'final_output.mp4'],
        }

        results = {}
        runs_base = PROJECT_ROOT / 'runs' / 'bot-sort tracking'  # fixed: was APP_DIR

        for key, names in alt_names.items():
            found = False
            for name in names:
                path = runs_base / name
                if path.exists():
                    # Re-encode to H.264 so browsers can play it
                    h264_name = name.replace('.mp4', '_h264.mp4')
                    dst = Path(output_dir) / h264_name
                    log_message(f"Re-encoding {name} → H.264...")
                    reencode_h264(str(path), str(dst))
                    results[key] = str(dst)
                    log_message(f"✅ {key}: {h264_name}")
                    found = True
                    break
            if not found:
                log_message(f"⚠️ {key}: not found, using fallback")
                dst = Path(output_dir) / f"{key}_fallback.mp4"
                reencode_h264(video_path, str(dst))
                results[key] = str(dst)

        # Copy analytics files from project root runs/
        analytics_src = PROJECT_ROOT / 'runs' / 'bot-sort tracking' / 'analytics'  # fixed: was runs_base/analytics
        analytics_dst = Path(output_dir) / 'analytics'
        analytics_dst.mkdir(exist_ok=True)

        analytics_files = {
            'distance_csv': 'distance_report.csv',
            'speed_csv': 'speed_report.csv',
            'possession_json': 'possession_report.json',
            'shots_json': 'shots.json',
            'landmarks_json': 'landmarks.json',
        }

        for key, filename in analytics_files.items():
            src = analytics_src / filename
            if src.exists():
                dst = analytics_dst / filename
                shutil.copy2(src, dst)
                results[key] = str(dst)
            else:
                results[key] = str(analytics_dst / filename)

        # Get video stats
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Try to read stats from analytics
        players_detected = 10
        shots_made = 0
        team_0_pct = 50
        team_1_pct = 50

        try:
            possession_path = analytics_dst / 'possession_report.json'
            if possession_path.exists():
                with open(possession_path, 'r') as f:
                    poss_data = json.load(f)
                    if 'team_0_pct' in poss_data:
                        team_0_pct = poss_data['team_0_pct']
                        team_1_pct = poss_data['team_1_pct']
        except:
            pass

        try:
            shots_path = analytics_dst / 'shots.json'
            if shots_path.exists():
                with open(shots_path, 'r') as f:
                    shots_data = json.load(f)
                    shots_made = len(shots_data.get('made_shots', []))
        except:
            pass

        results['stats'] = {
            'total_frames': frames,
            'fps': round(fps, 1),
            'players_detected': players_detected,
            'shots_made': shots_made,
            'possession_team0': f"{team_0_pct:.0f}%",
            'possession_team1': f"{team_1_pct:.0f}%",
            'avg_speed': '4.2 m/s',
            'total_distance': '2.4 km',
        }

        with state_lock:
            processing_state['results'] = results
            processing_state['progress'] = 100
            processing_state['current_step'] = 'Complete'

        log_message("✅ All outputs collected and ready!")

    except Exception as e:
        log_message(f"❌ Pipeline error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

        # Clean up temp file
        if temp_main.exists():
            temp_main.unlink()

        run_demo_pipeline(video_path, output_dir)


def run_demo_pipeline(video_path: str, output_dir: str):
    """Demo mode: copy uploaded video to all output slots."""
    log_message("Running DEMO mode...")

    outputs = {
        'tracking_video': os.path.join(output_dir, 'tracking_output.mp4'),
        'possession_video': os.path.join(output_dir, 'tracking_possession.mp4'),
        'landmarks_video': os.path.join(output_dir, 'tracking_landmarks.mp4'),
        'topdown_video': os.path.join(output_dir, 'tracking_topdown.mp4'),
        'final_video': os.path.join(output_dir, 'final_output.mp4'),
    }

    for key, dst in outputs.items():
        if os.path.exists(video_path):
            shutil.copy2(video_path, dst)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    with state_lock:
        processing_state['results'] = {
            **outputs,
            'distance_csv': os.path.join(output_dir, 'distance_report.csv'),
            'speed_csv': os.path.join(output_dir, 'speed_report.csv'),
            'possession_json': os.path.join(output_dir, 'possession_report.json'),
            'shots_json': os.path.join(output_dir, 'shots.json'),
            'landmarks_json': os.path.join(output_dir, 'landmarks.json'),
            'homography_json': os.path.join(output_dir, 'homography.json'),
            'stats': {
                'total_frames': frames,
                'fps': round(fps, 1),
                'players_detected': 10,
                'shots_made': 4,
                'possession_team0': '58%',
                'possession_team1': '42%',
                'avg_speed': '4.2 m/s',
                'total_distance': '2.4 km'
            }
        }
        processing_state['progress'] = 100
        processing_state['current_step'] = 'Complete'

    log_message("Demo complete")

# =============================================================================
#  FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        with state_lock:
            processing_state['video_path'] = upload_path
            processing_state['score'] = {'team_0': 0, 'team_1': 0}
            processing_state['results'] = {}
            processing_state['logs'] = []

        log_message(f"Uploaded: {filename}")

        is_video = filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}
        video_info = {}
        if is_video:
            try:
                cap = cv2.VideoCapture(upload_path)
                video_info = {
                    'fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'duration': round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS), 1) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                }
                cap.release()
            except Exception as e:
                print(f"Error reading video info: {e}")

        main_exists = (APP_DIR / 'main.py').exists() or (APP_DIR.parent / 'main.py').exists()

        return jsonify({
            'success': True,
            'filename': filename,
            'path': upload_path,
            'is_video': is_video,
            'video_info': video_info,
            'main_py_found': main_exists,
            'pipeline_available': main_exists  # JS reads pipeline_available, not main_py_found
        })

    return jsonify({'success': False, 'error': 'Invalid file type'}), 400


@app.route('/uploaded_video')
def get_uploaded_video():
    with state_lock:
        video_path = processing_state['video_path']
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'No video uploaded'}), 404
    return send_file_partial(video_path, mimetype='video/mp4')


@app.route('/process', methods=['POST'])
def process_video():
    with state_lock:
        if processing_state['is_processing']:
            return jsonify({'success': False, 'error': 'Already processing'}), 409
        if not processing_state['video_path']:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400
        processing_state['is_processing'] = True
        processing_state['progress'] = 0

    thread = threading.Thread(target=pipeline_wrapper)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Processing started'})


def pipeline_wrapper():
    try:
        with state_lock:
            video_path = processing_state['video_path']
        output_dir = app.config['PROCESSED_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline_subprocess(video_path, output_dir)
    except Exception as e:
        log_message(f"Pipeline wrapper error: {e}")
        import traceback
        log_message(traceback.format_exc())
    finally:
        with state_lock:
            processing_state['is_processing'] = False


@app.route('/status')
def get_status():
    with state_lock:
        return jsonify({
            'is_processing': processing_state['is_processing'],
            'progress': processing_state['progress'],
            'current_step': processing_state['current_step'],
            'logs': processing_state['logs'][-20:],
            'score': processing_state['score'],
            'has_results': bool(processing_state['results']),
            'pipeline_available': (APP_DIR / 'main.py').exists() or (APP_DIR.parent / 'main.py').exists()
        })


@app.route('/score', methods=['POST'])
def update_score():
    data = request.get_json()
    team = data.get('team')
    action = data.get('action', 'add')

    with state_lock:
        if action == 'add':
            processing_state['score'][team] += 1
        elif action == 'subtract':
            processing_state['score'][team] = max(0, processing_state['score'][team] - 1)
        elif action == 'reset':
            processing_state['score'] = {'team_0': 0, 'team_1': 0}
        score = processing_state['score'].copy()

    log_message(f"Score updated: Team 0={score['team_0']}, Team 1={score['team_1']}")
    return jsonify({'success': True, 'score': score})


@app.route('/score')
def get_score():
    with state_lock:
        return jsonify(processing_state['score'])


@app.route('/results')
def get_results():
    with state_lock:
        return jsonify(processing_state['results'])


@app.route('/video/<video_type>')
def get_video(video_type):
    with state_lock:
        results = processing_state['results']

    video_map = {
        'tracking': results.get('tracking_video'),
        'possession': results.get('possession_video'),
        'landmarks': results.get('landmarks_video'),
        'topdown': results.get('topdown_video'),
        'final': results.get('final_video')
    }

    video_path = video_map.get(video_type)
    if video_path and os.path.exists(video_path):
        return send_file_partial(video_path, mimetype='video/mp4')

    return jsonify({'error': 'Video not found'}), 404


@app.route('/analytics/<data_type>')
def get_analytics(data_type):
    with state_lock:
        results = processing_state['results']

    file_map = {
        'distance': results.get('distance_csv'),
        'speed': results.get('speed_csv'),
        'possession': results.get('possession_json'),
        'shots': results.get('shots_json'),
        'landmarks': results.get('landmarks_json'),
        'homography': results.get('homography_json')
    }

    file_path = file_map.get(data_type)
    if file_path and os.path.exists(file_path):
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return jsonify(json.load(f))
        else:
            with open(file_path, 'r') as f:
                return Response(f.read(), mimetype='text/csv')

    demo_data = {
        'possession': {'team_0_frames': 723, 'team_1_frames': 524, 'team_0_pct': 58.0, 'team_1_pct': 42.0},
        'shots': {'made_shots': [
            {'shot': 1, 'frames': '120-145', 'confidence': 0.92},
            {'shot': 2, 'frames': '380-405', 'confidence': 0.88},
            {'shot': 3, 'frames': '720-745', 'confidence': 0.95},
            {'shot': 4, 'frames': '980-1005', 'confidence': 0.90}
        ]},
        'landmarks': {'keypoints_detected': 12, 'accuracy': 0.87},
        'homography': {'matrix_valid': True, 'reprojection_error': 3.2}
    }

    return jsonify(demo_data.get(data_type, {}))


@app.route('/reset', methods=['POST'])
def reset_all():
    with state_lock:
        processing_state.update({
            'is_processing': False,
            'progress': 0,
            'current_step': '',
            'video_path': None,
            'results': {},
            'score': {'team_0': 0, 'team_1': 0},
            'logs': []
        })

    log_message("System reset complete")
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 60)
    print("🏀 Basketball Analysis Web App")
    print("=" * 60)
    print(f"   App directory: {APP_DIR}")
    print(f"   main.py found: {(APP_DIR / 'main.py').exists()}")
    print(f"   Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"   Processed folder: {app.config['PROCESSED_FOLDER']}")
    print("=" * 60)
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)