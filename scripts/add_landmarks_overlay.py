"""
scripts/add_landmarks_overlay.py
─────────────────────────────────
Post-processing pass that overlays court landmarks on a rendered video.

Takes any video already produced by main.py (with bboxes, IDs, team
colors, and stats already drawn) and adds the Roboflow court keypoint
detections as a final visual layer.

This is intentionally a SEPARATE pass instead of being merged into
main.py for three reasons:
  1. main.py already runs YOLO + BoT-SORT + ReID + CLIP per frame.
     Adding another model inside that loop slows it and risks failures
     killing tracking output.
  2. The keypoint layer is purely visual — no downstream code reads it.
     Decoupling means landmarks can be re-rendered without re-running
     the whole tracking pipeline.
  3. If the Roboflow API is unreachable, the existing final video is
     still produced and saved.

Usage
─────
  Set ROBOFLOW_API_KEY then:

  python scripts/add_landmarks_overlay.py \
      --input  "runs/bot-sort tracking/final_output.mp4" \
      --output "runs/bot-sort tracking/final_with_landmarks.mp4"

Optional flags
──────────────
  --conf 0.25         Lower threshold to see more keypoints
  --skip 2            Run inference every 2nd frame (cheaper, copies
                      keypoints between). Use only if camera moves slowly.
  --max-frames 100    Cap output length for quick previews
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

# Make the project's src/ importable regardless of where the script is run
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analytics.court_detection import CourtKeypointDetector, draw_keypoints


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",      required=True, help="Rendered video to enhance")
    ap.add_argument("--output",     required=True, help="Output MP4 path")
    ap.add_argument("--api-key",    default=None,
                    help="Roboflow API key (defaults to $ROBOFLOW_API_KEY)")
    ap.add_argument("--model-id",   default="basketball-court-detection-2/13")
    ap.add_argument("--conf",       type=float, default=0.30)
    ap.add_argument("--skip",       type=int,   default=1,
                    help="Run inference every Nth frame (1 = every frame)")
    ap.add_argument("--max-frames", type=int,   default=0,
                    help="Cap output length (0 = full video)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}", file=sys.stderr)
        return 2

    # Build the detector (downloads weights on first run, then cached)
    print(f"[Overlay] Loading court model: {args.model_id}")
    try:
        detector = CourtKeypointDetector(
            model_id       = args.model_id,
            api_key        = args.api_key,
            conf_threshold = args.conf,
        )
    except (ImportError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"[Overlay] Model ready.")

    # ── Video I/O ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.input}", file=sys.stderr)
        return 2

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(args.max_frames, n_total) if args.max_frames > 0 else n_total

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    if not writer.isOpened():
        print(f"ERROR: Cannot open writer for {args.output}", file=sys.stderr)
        return 2

    print(f"[Overlay] Input:  {args.input}")
    print(f"[Overlay]         {W}x{H} @ {fps:.1f} fps, {n_total} frames")
    print(f"[Overlay] Output: {args.output}")
    print(f"[Overlay] Conf:   ≥ {args.conf}    Skip: every {args.skip} frame(s)")

    # ── Loop ─────────────────────────────────────────────────────────────────
    cached_kps: list = []
    n_with_kp = 0
    n_inferred = 0
    t0 = time.time()

    try:
        for fidx in range(n_target):
            ok, frame = cap.read()
            if not ok:
                break

            # Run inference every Nth frame; reuse cache otherwise
            if fidx % args.skip == 0:
                try:
                    cached_kps = detector.detect(frame)
                    n_inferred += 1
                except Exception as e:
                    print(f"  [warn] frame {fidx}: inference failed ({e}); reusing cache")

            if cached_kps:
                n_with_kp += 1
                draw_keypoints(frame, cached_kps)

            writer.write(frame)

            if (fidx + 1) % 30 == 0:
                elapsed = time.time() - t0
                rate = (fidx + 1) / elapsed
                eta = (n_target - fidx - 1) / max(1e-3, rate)
                kp_pct = 100 * n_with_kp / (fidx + 1)
                print(f"  frame {fidx+1}/{n_target}  "
                      f"({rate:.1f} fps, ETA {eta:.0f}s)  "
                      f"kp_rate={kp_pct:.0f}%")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    kp_pct = 100 * n_with_kp / max(1, n_target)
    print(f"\n[Overlay] ✓ Done. {n_target} frames in {elapsed:.1f}s")
    print(f"[Overlay]   Inferences run:        {n_inferred}")
    print(f"[Overlay]   Frames with keypoints: {n_with_kp}/{n_target} ({kp_pct:.1f}%)")
    print(f"[Overlay]   Output:                {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
