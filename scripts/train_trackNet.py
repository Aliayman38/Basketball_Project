"""
scripts/train_tracknet.py
──────────────────────────
Train TrackNetV2 on your basketball video(s).

How it works
────────────
1. We extract every frame from the video(s).
2. For each frame we look up the ball (cx, cy) from your YOLO labels
   (the .txt files in data/dataset/train/labels/).
3. We create sliding windows of 3 consecutive frames → stacked 9-ch tensor.
4. We generate a Gaussian heatmap target from (cx, cy).
5. We train TrackNet to predict that heatmap.

You do NOT need to relabel anything. Your existing Roboflow YOLO labels
already contain ball cx/cy — we just convert them.

Run
───
    python scripts/train_tracknet.py \
        --video-dir  data/videos/          \   ← folder with your .mp4 files
        --label-dir  data/dataset/train/labels/ \  ← YOLO .txt label folder
        --image-dir  data/dataset/train/images/ \  ← matching images folder
        --epochs 30 --batch 4 --save-path weights/tracknet_best.pt

Or train directly from video frames (no pre-extracted images needed):
    python scripts/train_tracknet.py \
        --video game1.mp4 game2.mp4 \
        --epochs 30
"""

from __future__ import annotations
import argparse
import os
import sys
import glob
import json
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from detection.tracknet import TrackNet, make_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_BALL   = 0
INPUT_W      = 640
INPUT_H      = 352
SIGMA        = 5.0        # Gaussian heatmap spread (pixels at INPUT resolution)
HEATMAP_THR  = 0.5        # minimum heatmap peak to count as detection


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrackNetDataset(Dataset):
    """
    Sliding-window dataset of 3 consecutive frames.

    Each sample:
        x : (9, H, W)  float32, normalised 0..1
        y : (1, H, W)  float32, Gaussian heatmap 0..1
        has_ball : bool  — False when frame has no ball label
    """

    def __init__(
        self,
        sequences: list[list[dict]],   # list of clips; each clip = list of {frame, cx, cy}
        img_w: int = INPUT_W,
        img_h: int = INPUT_H,
        sigma: float = SIGMA,
        augment: bool = True,
    ):
        self.img_w   = img_w
        self.img_h   = img_h
        self.sigma   = sigma
        self.augment = augment

        # Build list of (clip_idx, start_frame) triplets
        self.samples: list[tuple[int, int]] = []
        self.sequences = sequences

        for ci, clip in enumerate(sequences):
            for fi in range(len(clip) - 2):
                # Only include triplets where the MIDDLE frame has a ball
                if clip[fi + 1]["has_ball"]:
                    self.samples.append((ci, fi))

        print(f"[Dataset] {len(self.samples)} valid triplets from "
              f"{len(sequences)} clips")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ci, fi = self.samples[idx]
        clip   = self.sequences[ci]

        frames = []
        for offset in range(3):
            frame_data = clip[fi + offset]
            img = self._load_and_resize(frame_data["frame_path"])
            frames.append(img)

        # Target: heatmap for the MIDDLE frame
        mid = clip[fi + 1]
        cx_scaled = mid["cx"] * self.img_w
        cy_scaled = mid["cy"] * self.img_h

        target = make_heatmap(cx_scaled, cy_scaled, self.img_h, self.img_w, self.sigma)

        # Stack 3 frames → (9, H, W)
        x = torch.cat([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in frames
        ], dim=0)   # (9, H, W)

        # Augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            x      = torch.flip(x, dims=[2])
            target = torch.flip(target, dims=[2])

        return x, target

    def _load_and_resize(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.resize(img, (self.img_w, self.img_h))


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_label(label_path: str) -> tuple[float, float] | None:
    """
    Parse a YOLO .txt label file and return ball (cx_norm, cy_norm) or None.
    YOLO format: class cx cy w h  (all normalised 0..1)
    """
    if not os.path.exists(label_path):
        return None

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) == CLASS_BALL:
                return float(parts[1]), float(parts[2])   # cx, cy normalised

    return None


def build_sequences_from_images(
    image_dir: str,
    label_dir: str,
) -> list[list[dict]]:
    """
    Build one clip from sorted image files + their YOLO labels.
    All images are treated as one continuous sequence.
    """
    exts  = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    clip = []
    missing = 0
    for p in paths:
        stem  = Path(p).stem
        label = os.path.join(label_dir, stem + ".txt")
        ball  = parse_yolo_label(label)
        clip.append({
            "frame_path": p,
            "has_ball":   ball is not None,
            "cx":         ball[0] if ball else 0.5,
            "cy":         ball[1] if ball else 0.5,
        })
        if ball is None:
            missing += 1

    print(f"[Loader] {len(clip)} frames, {missing} missing ball labels "
          f"({100*missing/max(len(clip),1):.1f}%)")
    return [clip]


def build_sequences_from_videos(
    video_paths: list[str],
    save_dir: str = "data/tracknet_frames",
) -> list[list[dict]]:
    """
    Extract frames from videos and use YOLO (YOLOv11L) to auto-label the ball.
    This lets you train TrackNet even without pre-extracted image datasets.
    """
    from ultralytics import YOLO

    os.makedirs(save_dir, exist_ok=True)
    print(f"[AutoLabel] Loading YOLOv11 for ball auto-labelling…")
    yolo = YOLO("yolo11l.pt")   # or your fine-tuned best.pt

    all_sequences = []

    for vid_path in video_paths:
        print(f"[AutoLabel] Processing: {vid_path}")
        cap     = cv2.VideoCapture(vid_path)
        vid_name = Path(vid_path).stem
        clip_dir = os.path.join(save_dir, vid_name)
        os.makedirs(clip_dir, exist_ok=True)

        clip    = []
        fidx    = 0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Save frame
            frame_path = os.path.join(clip_dir, f"{fidx:06d}.jpg")
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)

            # Auto-label ball with YOLO
            res   = yolo(frame, conf=0.10, classes=[CLASS_BALL],
                         imgsz=1280, verbose=False)[0]
            ball  = None
            if res.boxes is not None and len(res.boxes):
                box  = res.boxes.xyxy[0].cpu().numpy()
                h, w = frame.shape[:2]
                cx   = float((box[0] + box[2]) / 2 / w)
                cy   = float((box[1] + box[3]) / 2 / h)
                ball = (cx, cy)

            clip.append({
                "frame_path": frame_path,
                "has_ball":   ball is not None,
                "cx":         ball[0] if ball else 0.5,
                "cy":         ball[1] if ball else 0.5,
            })

            fidx += 1
            if fidx % 200 == 0:
                print(f"  {fidx}/{total} frames processed")

        cap.release()
        all_sequences.append(clip)
        print(f"[AutoLabel] {vid_name}: {fidx} frames, "
              f"{sum(1 for f in clip if f['has_ball'])} with ball")

    return all_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalMSELoss(nn.Module):
    """
    Weighted MSE that penalises false negatives harder than false positives.

    Standard MSE treats all pixels equally — the ball occupies ~0.01% of
    the image, so MSE loss is dominated by the background.

    FocalMSE multiplies the loss at ball-region pixels by `pos_weight`
    to force the network to care about getting the peak right.
    """

    def __init__(self, pos_weight: float = 100.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Weight map: high near ball centre, 1.0 everywhere else
        weight = 1.0 + (self.pos_weight - 1.0) * target
        loss   = (weight * (pred - target) ** 2).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def detection_accuracy(
    pred_maps:   torch.Tensor,
    target_maps: torch.Tensor,
    threshold:   float = HEATMAP_THR,
    tolerance_px: float = 10.0,
    input_w: int = INPUT_W,
    input_h: int = INPUT_H,
) -> dict:
    """
    Compute TP, FP, FN and accuracy for a batch of heatmaps.

    A prediction is TP if:
      - Predicted peak > threshold  AND
      - Euclidean distance to target peak ≤ tolerance_px
    """
    from detection.tracknet import heatmap_to_point

    tp = fp = fn = 0

    for pred, tgt in zip(pred_maps, target_maps):
        pred_pt = heatmap_to_point(pred.squeeze(0), threshold)
        tgt_pt  = heatmap_to_point(tgt.squeeze(0),  threshold * 0.3)

        if tgt_pt is None:
            if pred_pt is not None:
                fp += 1
            continue

        if pred_pt is None:
            fn += 1
            continue

        dist = np.hypot(pred_pt[0] - tgt_pt[0], pred_pt[1] - tgt_pt[1])
        if dist <= tolerance_px:
            tp += 1
        else:
            fp += 1

    total      = tp + fp + fn
    accuracy   = tp / max(total, 1)
    precision  = tp / max(tp + fp, 1)
    recall     = tp / max(tp + fn, 1)
    f1         = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"acc": accuracy, "prec": precision, "rec": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  TrackNetV2 Training")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Save    : {args.save_path}\n")

    # ── Build dataset ─────────────────────────────────────────────────────
    if args.video:
        sequences = build_sequences_from_videos(args.video)
    elif args.image_dir and args.label_dir:
        sequences = build_sequences_from_images(args.image_dir, args.label_dir)
    else:
        print("[Train] ERROR: provide --video OR (--image-dir AND --label-dir)")
        sys.exit(1)

    # Train/val split (90/10 on clips)
    random.shuffle(sequences)
    n_val      = max(1, int(len(sequences) * 0.1))
    val_seqs   = sequences[:n_val]
    train_seqs = sequences[n_val:]

    # If only one clip, split by frames
    if len(sequences) == 1:
        clip   = sequences[0]
        n_val  = max(10, int(len(clip) * 0.1))
        val_seqs   = [clip[-n_val:]]
        train_seqs = [clip[:-n_val]]

    train_ds = TrackNetDataset(train_seqs, augment=True)
    val_ds   = TrackNetDataset(val_seqs,   augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = TrackNet(in_frames=3).to(device)
    print(f"[Model] {model}")

    if args.resume and os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Train] Resumed from {args.save_path}")

    criterion = FocalMSELoss(pos_weight=100.0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1      = 0.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    history: list[dict] = []

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_dl), 1)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []

        with torch.no_grad():
            for x, y in val_dl:
                x, y  = x.to(device), y.to(device)
                pred  = model(x)
                val_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu())
                all_tgts.append(y.cpu())

        val_loss  /= max(len(val_dl), 1)
        preds_cat  = torch.cat(all_preds)
        tgts_cat   = torch.cat(all_tgts)
        metrics    = detection_accuracy(preds_cat, tgts_cat)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"F1={metrics['f1']:.4f}  Recall={metrics['rec']:.4f}  "
            f"Prec={metrics['prec']:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **metrics})

        # Save best checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "f1":      best_f1,
                "metrics": metrics,
                "input_w": INPUT_W,
                "input_h": INPUT_H,
            }, args.save_path)
            print(f"  ✓ Saved best model (F1={best_f1:.4f}) → {args.save_path}")

    # Save training history
    hist_path = args.save_path.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Best F1: {best_f1:.4f}")
    print(f"[Train] Weights : {args.save_path}")
    print(f"[Train] History : {hist_path}")
    print(f"\nNext step:")
    print(f"  Update config/config.yaml:")
    print(f"    tracknet_weights: {args.save_path}")
    print(f"  Then run main.py — BallTracker will use TrackNet automatically.\n")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TrackNetV2 for ball tracking")
    p.add_argument("--video",      nargs="+", default=None,
                   help="One or more video files (auto-labels ball with YOLO)")
    p.add_argument("--image-dir",  default=None,
                   help="Folder of pre-extracted images")
    p.add_argument("--label-dir",  default=None,
                   help="YOLO .txt label folder matching --image-dir")
    p.add_argument("--save-path",  default="weights/tracknet_best.pt")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())