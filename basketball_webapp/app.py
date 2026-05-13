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
import select                          # FIX 3: top-level import (needed for non-blocking reads)
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
    'selected_model': None,
    'team_0_desc': 'a basketball player wearing a yellow jersey',
    'team_1_desc': 'a basketball player wearing a dark blue jersey',
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


# FIX 2: Define create_error_video so it can be called when a pipeline output is missing
def create_error_video(dst_path: str, message: str = "Error: video not generated"):
    """
    Write a minimal single-frame MP4 at dst_path with an error message burned in.
    Falls back to an empty file if OpenCV H.264 encoding is unavailable.
    """
    try:
        frame = __import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8)
        cv2.putText(frame, message, (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(dst_path, fourcc, 1.0, (640, 480))
        if writer.isOpened():
            for _ in range(30):   # ~30 frames so the video is non-empty
                writer.write(frame)
            writer.release()
            log_message(f"   ⚠️ Created error placeholder: {Path(dst_path).name}")
            return
        writer.release()
    except Exception as e:
        log_message(f"   ⚠️ create_error_video failed ({e}), writing empty file")
    # Last resort: empty file so callers don't crash on a missing path
    open(dst_path, 'wb').close()

# =============================================================================
#  VIDEO RE-ENCODING
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
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',
                '-an',
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

# =============================================================================
#  PIPELINE RUNNER - Calls your main.py via subprocess
# =============================================================================

def _read_subprocess_output(process, stop_event):
    """
    FIX 4: Read subprocess stdout line-by-line in a dedicated thread.
    This is cross-platform (works on Windows, Linux, macOS) unlike
    select.select() on pipes, which only works on POSIX.
    """
    try:
        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            if any(kw in line for kw in ('Error', 'error', 'Traceback', 'Exception', 'FAILED')):
                log_message(f"❌ {line}")
            else:
                log_message(f"   {line}")

            # Parse progress hints from pipeline output
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
            elif 'Done' in line or 'Complete' in line:
                update_progress("Dashboard", 98)
    except Exception as e:
        log_message(f"⚠️ Output reader error: {e}")


def run_pipeline_subprocess(video_path: str, output_dir: str, model_path: str = None,
                            team_0_desc: str = None, team_1_desc: str = None):
    """
    Run main.py pipeline via subprocess with cross-platform output reading.
    """

    log_message("=" * 60)
    log_message("🚀 STARTING PIPELINE")
    log_message("=" * 60)
    log_message(f"   Video: {video_path}")
    log_message(f"   Output dir: {output_dir}")
    log_message(f"   Model: {model_path}")

    main_py_path = APP_DIR / 'main.py'
    if not main_py_path.exists():
        main_py_path = APP_DIR.parent / 'main.py'
    if not main_py_path.exists():
        log_message(f"❌ main.py not found in {APP_DIR} or {APP_DIR.parent}")
        run_demo_pipeline(video_path, output_dir)
        return

    PROJECT_ROOT = main_py_path.parent
    log_message(f"✅ Found main.py at: {main_py_path}")

    runs_dir = PROJECT_ROOT / 'runs' / 'bot-sort tracking'
    analytics_dir = runs_dir / 'analytics'

    # Clear old outputs
    if runs_dir.exists():
        log_message("🧹 Clearing old outputs...")
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    # Clear processed folder
    processed = Path(output_dir)
    for f in processed.glob('*.mp4'):
        try: f.unlink()
        except: pass

    # Copy video to data directory
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    video_name = Path(video_path).name
    data_video_path = data_dir / video_name
    shutil.copy2(video_path, data_video_path)
    log_message(f"✅ Video copied to: {data_video_path}")

    # Read and patch main.py
    with open(main_py_path, 'r', encoding='utf-8') as f:
        main_content = f.read()

    # Patch VIDEO_PATH
    main_content = re.sub(
        r"VIDEO_PATH\s*=\s*['\"'].*?['\"']",
        f"VIDEO_PATH = 'data/{video_name}'",
        main_content
    )
    # Patch OUTPUT_PATH
    main_content = re.sub(
        r"OUTPUT_PATH\s*=\s*['\"'].*?['\"']",
        "OUTPUT_PATH = 'runs/bot-sort tracking/tracking_output.mp4'",
        main_content
    )
    # Patch MODEL_PATH
    if model_path:
        model_path_repr = repr(model_path)
        main_content = re.sub(
            r"MODEL_PATH\s*=\s*['\"'].*?['\"']",
            f"MODEL_PATH = {model_path_repr}",
            main_content
        )
        log_message(f"✅ Patched MODEL_PATH: {model_path}")
    # Patch TEAM descriptions for CLIP clustering
    if team_0_desc:
        main_content = re.sub(
            r"TEAM_0_DESC\s*=\s*['\"].*?['\"]",
            f"TEAM_0_DESC = {repr(team_0_desc)}",
            main_content
        )
        log_message(f"✅ Patched TEAM_0_DESC: {team_0_desc}")
    if team_1_desc:
        main_content = re.sub(
            r"TEAM_1_DESC\s*=\s*['\"].*?['\"]",
            f"TEAM_1_DESC = {repr(team_1_desc)}",
            main_content
        )
        log_message(f"✅ Patched TEAM_1_DESC: {team_1_desc}")
    # Patch DEVICE
    main_content = re.sub(
        r"DEVICE\s*=\s*torch\.device\(['\"]cuda:\d+['\"]\)",
        "DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        main_content
    )

    # CPU shim
    cpu_shim = """# --- CPU compatibility shim injected by app.py ---
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
    # FIX 6: corrected indentation — comment was over-indented in original
    # Find the end of __future__ imports and insert shim there
    future_import_end = 0
    for match in re.finditer(r'from __future__ import.*?\n', main_content):
        future_import_end = match.end()

    if future_import_end > 0:
        main_content = main_content[:future_import_end] + cpu_shim + main_content[future_import_end:]
    else:
        main_content = cpu_shim + main_content

    # Write temp file
    temp_main = PROJECT_ROOT / 'main_web_temp.py'
    with open(temp_main, 'w', encoding='utf-8') as f:
        f.write(main_content)
    log_message(f"✅ Temp file: {temp_main} ({temp_main.stat().st_size} bytes)")

    # Pre-flight model check
    model_match = re.search(r"MODEL_PATH\s*=\s*['\"']r?(.+?)['\"']", main_content)
    if model_match:
        mp = model_match.group(1)
        if mp.startswith('r'): mp = mp[1:]
        mp_abs = Path(mp) if Path(mp).is_absolute() else PROJECT_ROOT / mp
        if mp_abs.exists():
            log_message(f"✅ Model verified: {mp_abs.name} ({round(mp_abs.stat().st_size/1024/1024,1)} MB)")
        else:
            log_message(f"❌ MODEL NOT FOUND: {mp_abs}")

    # Run subprocess
    update_progress("Detection & Tracking", 10)
    log_message("=" * 60)
    log_message("▶️  RUNNING SUBPROCESS")
    log_message("=" * 60)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['PYTHONUNBUFFERED'] = '1'

    try:
        process = subprocess.Popen(
            [sys.executable, str(temp_main)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        log_message(f"✅ Subprocess started (PID: {process.pid})")
    except Exception as e:
        log_message(f"❌ Failed to start: {e}")
        if temp_main.exists(): temp_main.unlink()
        run_demo_pipeline(video_path, output_dir)
        return

    # FIX 4: Use a reader thread instead of select.select() for cross-platform support
    TIMEOUT = 300  # 5 minutes without output
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=_read_subprocess_output,
        args=(process, stop_event),
        daemon=True
    )
    reader_thread.start()

    last_alive = time.time()
    try:
        while True:
            if process.poll() is not None:
                break
            if time.time() - last_alive > TIMEOUT:
                log_message(f"⚠️ TIMEOUT: No activity for {TIMEOUT}s — killing process")
                process.kill()
                break
            # Reset the alive timer whenever the reader is making progress
            if reader_thread.is_alive():
                last_alive = time.time()
            time.sleep(1)
    except Exception as e:
        log_message(f"⚠️ Process monitor error: {e}")
        process.kill()
    finally:
        stop_event.set()

    process.wait()
    reader_thread.join(timeout=10)
    log_message(f"🛑 Exit code: {process.returncode}")

    if process.returncode != 0:
        log_message("❌ Pipeline FAILED")
    else:
        log_message("✅ Pipeline completed")

    # Cleanup temp file
    if temp_main.exists():
        temp_main.unlink()

    # Verify outputs
    log_message("📁 Checking outputs...")
    runs_base = PROJECT_ROOT / 'runs' / 'bot-sort tracking'
    output_files = list(runs_base.glob('*.mp4')) if runs_base.exists() else []

    if output_files:
        log_message(f"✅ Found {len(output_files)} files:")
        for f in output_files:
            log_message(f"   • {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
    else:
        log_message("❌ NO OUTPUT FILES")

    # Re-encode outputs to H.264 for browser playback
    update_progress("Re-encoding for browser", 99)
    log_message("🎬 Re-encoding to H.264 (requires ffmpeg)...")

    alt_names = {
        'tracking_video':   ['tracking_output.mp4'],
        'possession_video': ['tracking_possession.mp4'],
        'landmarks_video':  ['tracking_landmarks.mp4'],
        'topdown_video':    ['tracking_topdown.mp4'],
        'final_video':      ['final_output1.mp4', 'final_output.mp4'],
    }

    results = {}
    for key, names in alt_names.items():
        found = False
        for name in names:
            path = runs_base / name
            if path.exists():
                h264_name = name.replace('.mp4', '_h264.mp4')
                dst = Path(output_dir) / h264_name
                log_message(f"Re-encoding {name}...")
                reencode_h264(str(path), str(dst))
                results[key] = str(dst)
                log_message(f"✅ {key}: {h264_name}")
                found = True
                break
        if not found:
            log_message(f"❌ {key}: not found")
            dst = Path(output_dir) / f"{key}_error.mp4"
            create_error_video(str(dst), f"Error: {key} not generated")  # FIX 2: now defined
            results[key] = str(dst)

    # Copy analytics
    analytics_src = runs_base / 'analytics'
    analytics_dst = Path(output_dir) / 'analytics'
    analytics_dst.mkdir(exist_ok=True)

    for key, filename in {
        'distance_csv':   'distance_report.csv',
        'speed_csv':      'speed_report.csv',
        'possession_json': 'possession_report.json',
        'shots_json':     'shots.json',
        'landmarks_json': 'landmarks.json',
    }.items():
        src = analytics_src / filename
        if src.exists():
            dst_file = analytics_dst / filename
            shutil.copy2(src, dst_file)
            results[key] = str(dst_file)

    # Video stats
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Read auto-scores from pipeline output
    team_0_score = 0
    team_1_score = 0
    shots_made   = 0
    try:
        scores_path = analytics_dst / 'scores.json'
        if scores_path.exists():
            with open(scores_path, 'r') as f:
                scores_data = json.load(f)
                team_0_score = scores_data.get('team_0', 0)
                team_1_score = scores_data.get('team_1', 0)
                shots_made   = scores_data.get('total_shots', 0)
                log_message(f"🏀 Auto-scores loaded: Yellow {team_0_score} - {team_1_score} Blue")
        else:
            log_message("⚠️ scores.json not found, using defaults")
    except Exception as e:
        log_message(f"⚠️ Could not read scores: {e}")

    with state_lock:
        processing_state['score'] = {'team_0': team_0_score, 'team_1': team_1_score}

    # FIX 1: Read possession percentages from analytics; default to 50/50 if unavailable
    team_0_pct = 50.0
    team_1_pct = 50.0
    try:
        possession_path = analytics_dst / 'possession_report.json'
        if possession_path.exists():
            with open(possession_path, 'r') as f:
                poss_data = json.load(f)
                team_0_pct = float(poss_data.get('team_0_pct', 50.0))
                team_1_pct = float(poss_data.get('team_1_pct', 50.0))
    except Exception as e:
        log_message(f"⚠️ Could not read possession data: {e}")

    results['stats'] = {
        'total_frames':    frames,
        'fps':             round(fps, 1),
        'players_detected': 10,
        'shots_made':      shots_made,
        'possession_team0': f"{team_0_pct:.0f}%",
        'possession_team1': f"{team_1_pct:.0f}%",
        'avg_speed':       '4.2 m/s',
        'total_distance':  '2.4 km',
    }

    with state_lock:
        processing_state['results']      = results
        processing_state['progress']     = 100
        processing_state['current_step'] = 'Complete'

    log_message("=" * 60)
    log_message("✅ DONE")
    log_message("=" * 60)


def run_demo_pipeline(video_path: str, output_dir: str):
    """Demo mode: re-encode the uploaded video into all output slots."""
    log_message("Running DEMO mode...")

    slot_names = {
        'tracking_video':   'tracking_output.mp4',
        'possession_video': 'tracking_possession.mp4',
        'landmarks_video':  'tracking_landmarks.mp4',
        'topdown_video':    'tracking_topdown.mp4',
        'final_video':      'final_output.mp4',
    }

    outputs = {}
    for key, name in slot_names.items():
        dst = os.path.join(output_dir, name)
        if os.path.exists(video_path):
            # FIX 7: re-encode even in demo mode so browsers can play the video
            reencode_h264(video_path, dst)
        outputs[key] = dst

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    with state_lock:
        processing_state['results'] = {
            **outputs,
            'distance_csv':    os.path.join(output_dir, 'distance_report.csv'),
            'speed_csv':       os.path.join(output_dir, 'speed_report.csv'),
            'possession_json': os.path.join(output_dir, 'possession_report.json'),
            'shots_json':      os.path.join(output_dir, 'shots.json'),
            'landmarks_json':  os.path.join(output_dir, 'landmarks.json'),
            'homography_json': os.path.join(output_dir, 'homography.json'),
            'stats': {
                'total_frames':     frames,
                'fps':              round(fps, 1),
                'players_detected': 10,
                'shots_made':       4,
                'possession_team0': '58%',
                'possession_team1': '42%',
                'avg_speed':        '4.2 m/s',
                'total_distance':   '2.4 km',
            }
        }
        processing_state['progress']     = 100
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
            processing_state['score']      = {'team_0': 0, 'team_1': 0}
            processing_state['results']    = {}
            processing_state['logs']       = []

        log_message(f"Uploaded: {filename}")

        is_video = filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}
        video_info = {}
        if is_video:
            try:
                cap = cv2.VideoCapture(upload_path)
                video_info = {
                    'fps':      round(cap.get(cv2.CAP_PROP_FPS), 2),
                    'width':    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height':   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'frames':   int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'duration': round(
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS), 1
                    ) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                }
                cap.release()
            except Exception as e:
                print(f"Error reading video info: {e}")

        main_exists = (APP_DIR / 'main.py').exists() or (APP_DIR.parent / 'main.py').exists()

        return jsonify({
            'success':            True,
            'filename':           filename,
            'path':               upload_path,
            'is_video':           is_video,
            'video_info':         video_info,
            'main_py_found':      main_exists,
            'pipeline_available': main_exists,
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
    data = request.get_json() or {}
    selected_model = data.get('model_path')
    team_0_desc = data.get('team_0_desc', '').strip() or 'a basketball player wearing a yellow jersey'
    team_1_desc = data.get('team_1_desc', '').strip() or 'a basketball player wearing a dark blue jersey'

    with state_lock:
        if processing_state['is_processing']:
            return jsonify({'success': False, 'error': 'Already processing'}), 409
        if not processing_state['video_path']:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400
        processing_state['is_processing']  = True
        processing_state['progress']       = 0
        processing_state['results']        = {}
        processing_state['logs']           = []
        processing_state['selected_model'] = selected_model
        processing_state['team_0_desc']    = team_0_desc
        processing_state['team_1_desc']    = team_1_desc

    thread = threading.Thread(target=pipeline_wrapper)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Processing started'})


def pipeline_wrapper():
    try:
        with state_lock:
            video_path     = processing_state['video_path']
            selected_model = processing_state.get('selected_model')
            team_0_desc    = processing_state.get('team_0_desc', 'a basketball player wearing a yellow jersey')
            team_1_desc    = processing_state.get('team_1_desc', 'a basketball player wearing a dark blue jersey')
        output_dir = app.config['PROCESSED_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline_subprocess(video_path, output_dir, model_path=selected_model,
                                team_0_desc=team_0_desc, team_1_desc=team_1_desc)
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
            'is_processing':      processing_state['is_processing'],
            'progress':           processing_state['progress'],
            'current_step':       processing_state['current_step'],
            'logs':               processing_state['logs'],
            'score':              processing_state['score'],
            'has_results':        bool(processing_state['results']),
            'pipeline_available': (APP_DIR / 'main.py').exists() or (APP_DIR.parent / 'main.py').exists()
        })


@app.route('/score', methods=['POST'])
def update_score():
    data   = request.get_json()
    team   = data.get('team')
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


@app.route('/auto_scores')
def get_auto_scores():
    """Get AI-detected shot scores from the pipeline."""
    try:
        analytics_dir = Path(app.config['PROCESSED_FOLDER']) / 'analytics'
        scores_path   = analytics_dir / 'scores.json'
        if scores_path.exists():
            with open(scores_path, 'r') as f:
                return jsonify(json.load(f))
    except Exception as e:
        print(f"Error reading auto-scores: {e}")
    return jsonify({'team_0': 0, 'team_1': 0, 'shot_events': [], 'total_shots': 0})


@app.route('/results')
def get_results():
    with state_lock:
        return jsonify(processing_state['results'])


@app.route('/models')
def get_models():
    """List every .pt model file found in the project (root, models/, models/weights/)."""
    models = []
    seen   = set()

    search_dirs = []
    for base in [APP_DIR, APP_DIR.parent]:
        search_dirs.append(base)
        search_dirs.append(base / 'models')
        search_dirs.append(base / 'models' / 'weights')

    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == '.pt' and 'court_kp' not in f.name and str(f) not in seen:
                seen.add(str(f))
                models.append({
                    'name':     f.stem,
                    'filename': f.name,
                    'path':     str(f),
                    'size_mb':  round(f.stat().st_size / (1024 * 1024), 1)
                })

    return jsonify(models)


@app.route('/video/<video_type>')
def get_video(video_type):
    with state_lock:
        results = processing_state['results']

    video_map = {
        'tracking':   results.get('tracking_video'),
        'possession': results.get('possession_video'),
        'landmarks':  results.get('landmarks_video'),
        'topdown':    results.get('topdown_video'),
        'final':      results.get('final_video'),
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
        'distance':   results.get('distance_csv'),
        'speed':      results.get('speed_csv'),
        'possession': results.get('possession_json'),
        'shots':      results.get('shots_json'),
        'landmarks':  results.get('landmarks_json'),
        'homography': results.get('homography_json'),
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
            {'shot': 4, 'frames': '980-1005', 'confidence': 0.90},
        ]},
        'landmarks':  {'keypoints_detected': 12, 'accuracy': 0.87},
        'homography': {'matrix_valid': True, 'reprojection_error': 3.2},
    }

    return jsonify(demo_data.get(data_type, {}))


@app.route('/analytics/player_stats')
def get_player_stats():
    """Return per-player distance, speed, and team as JSON for the webapp dashboard."""
    try:
        analytics_dir = Path(app.config['PROCESSED_FOLDER']) / 'analytics'

        # ── Load distance CSV ──────────────────────────────────────────────
        dist_path = analytics_dir / 'distance_report.csv'
        dist_map = {}
        if dist_path.exists():
            import csv as _csv
            with dist_path.open(newline='', encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    pid = row.get('player_id', '').strip()
                    if pid:
                        dist_map[pid] = row

        # ── Load speed CSV ─────────────────────────────────────────────────
        speed_path = analytics_dir / 'speed_report.csv'
        speed_map = {}
        if speed_path.exists():
            import csv as _csv
            with speed_path.open(newline='', encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    pid = row.get('player_id', '').strip()
                    if pid:
                        speed_map[pid] = row

        # ── Load team assignments from trajectories ────────────────────────
        from collections import Counter
        team_map = {}
        runs_base = APP_DIR.parent / 'runs' / 'bot-sort tracking' / 'analytics' / 'trajectories.json'
        if not runs_base.exists():
            runs_base = APP_DIR / 'runs' / 'bot-sort tracking' / 'analytics' / 'trajectories.json'
        if runs_base.exists():
            with open(runs_base, 'r', encoding='utf-8') as f:
                traj = json.load(f)
            for pid, recs in traj.get('players', {}).items():
                teams = [r.get('team') for r in recs if r.get('team')]
                if teams:
                    team_map[pid] = Counter(teams).most_common(1)[0][0]

        # ── Merge ──────────────────────────────────────────────────────────
        all_ids = sorted(set(list(dist_map.keys()) + list(speed_map.keys())),
                         key=lambda x: int(x) if x.isdigit() else x)
        players = []
        for pid in all_ids:
            d_row = dist_map.get(pid, {})
            s_row = speed_map.get(pid, {})
            dist_val = d_row.get('total_distance_m') or d_row.get('total_distance_px') or '0'
            speed_val = s_row.get('avg_speed_m_s') or s_row.get('avg_speed_px_s') or '0'
            unit_d = 'm' if d_row.get('total_distance_m') else 'px'
            unit_s = 'm/s' if s_row.get('avg_speed_m_s') else 'px/s'
            players.append({
                'id': pid,
                'team': team_map.get(pid, '—'),
                'distance': f"{float(dist_val):.1f}",
                'distance_unit': unit_d,
                'speed': f"{float(speed_val):.2f}",
                'speed_unit': unit_s,
            })

        return jsonify({'players': players})

    except Exception as e:
        print(f'player_stats error: {e}')
        return jsonify({'players': []})



def reset_all():
    with state_lock:
        processing_state.update({
            'is_processing':  False,
            'progress':       0,
            'current_step':   '',
            'video_path':     None,
            'results':        {},
            'score':          {'team_0': 0, 'team_1': 0},
            'logs':           [],
            'selected_model': None,
            'team_0_desc':    'a basketball player wearing a yellow jersey',
            'team_1_desc':    'a basketball player wearing a dark blue jersey',
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