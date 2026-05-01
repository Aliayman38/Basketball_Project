"""
src/team_clustering/clusterer.py
─────────────────────────────────
Advanced Unsupervised team assignment via Foundation Models.
Utilizes: SigLIP (Vision Encoder) + UMAP (Dim Reduction) + K-Means (Clustering).

Dataset class map  (Roboflow basketball-players v11)
─────────────────────────────────────────────────────
  0 → Ball
  1 → Clock
  2 → Hoop
  3 → Overlay
  4 → Player   ← clustered here
  5 → Ref      ← detected separately; auto-assigned to TEAM_REF

Pipeline Architecture
───────────────────
1. Crop player torso to isolate the jersey.
2. Pass the crop through Google's SigLIP vision model to extract a 
   semantic embedding vector (robust to shadows, lighting, and occlusion).
3. Buffer these high-dimensional embeddings per track_id.
4. After warm_up_frames, apply UMAP to reduce the embeddings to 2D space.
5. Apply K-Means(k=2) on the 2D UMAP projections to assign Team A and Team B.
6. For new tracks appearing after warm-up, transform via UMAP and predict via K-Means.
"""

from __future__ import annotations

import cv2
import torch
import numpy as np
from collections import defaultdict
from PIL import Image

import umap.umap_ as umap
from sklearn.cluster import KMeans
from transformers import SiglipImageProcessor, SiglipVisionModel

# ── Dataset class IDs ─────────────────────────────────────────────────────────
CLASS_BALL    = 0
CLASS_CLOCK   = 1
CLASS_HOOP    = 2
CLASS_OVERLAY = 3
CLASS_PLAYER  = 4   
CLASS_REF     = 5   

# ── Team label constants ──────────────────────────────────────────────────────
TEAM_A       = 0   
TEAM_B       = 1   
TEAM_REF     = 2   
TEAM_UNKNOWN = -1  

# ── Display colours per team  (BGR for OpenCV) ────────────────────────────────
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    TEAM_A:       (235, 110,  40),   # vivid blue  — Team A
    TEAM_B:       ( 40, 200,  60),   # vivid green — Team B
    TEAM_REF:     ( 50,  50, 220),   # vivid red   — Referees
    TEAM_UNKNOWN: (160, 160, 160),   # grey        — not yet assigned
}

# ── Human-readable names ──────────────────────────────────────────────────────
TEAM_NAMES: dict[int, str] = {
    TEAM_A:       "Team A",
    TEAM_B:       "Team B",
    TEAM_REF:     "Referee",
    TEAM_UNKNOWN: "Unknown",
}

# ── Default torso slice ───────────────────────────────────────────────────────
_TORSO_TOP  = 0.15   # skip head and neck
_TORSO_BOT  = 0.50   # stop before shorts


# ─────────────────────────────────────────────────────────────────────────────
class TeamClusterer:
    """
    Assigns basketball player tracks to teams using SigLIP + UMAP + KMeans.
    """

    def __init__(
        self,
        warm_up_frames: int = 60,
        torso_ratio: tuple[float, float] = (_TORSO_TOP, _TORSO_BOT),
        min_obs: int = 5,
        device: str | None = None
    ) -> None:
        self.warm_up_frames = warm_up_frames
        self.torso_ratio = torso_ratio
        self.min_obs = min_obs

        # Set device dynamically
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[TeamClusterer] Initializing SigLIP on {self.device}...")
        
        # Load SigLIP Model and Processor
        model_id = "google/siglip-base-patch16-224"
        self.processor = SiglipImageProcessor.from_pretrained(model_id)
        self.siglip_model = SiglipVisionModel.from_pretrained(model_id).to(self.device)
        self.siglip_model.eval()

        # {track_id: [embedding_vector, ...]}
        self._embed_buffer: dict[int, list[np.ndarray]] = defaultdict(list)
        
        # {track_id: team_id}
        self._team_labels: dict[int, int] = {}

        # Dimensionality Reduction & Clustering Models
        self.umap_reducer = umap.UMAP(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

        self.is_fitted: bool = False
        self._frame_idx: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, tracked_dets: list[dict]) -> None:
        """
        Extracts embeddings for each player in the frame and manages clustering.
        """
        for det in tracked_dets:
            cid = int(det["class_id"])
            tid = int(det.get("track_id", -1))

            if tid == -1:
                continue

            if cid == CLASS_REF:
                self._team_labels[tid] = TEAM_REF
                continue

            if cid != CLASS_PLAYER:
                continue

            embedding = self._extract_embedding(frame, det["bbox"])
            if embedding is not None:
                self._embed_buffer[tid].append(embedding)

        self._frame_idx += 1

        if self._frame_idx == self.warm_up_frames and not self.is_fitted:
            self._fit()

        if self.is_fitted:
            self._assign_pending()

    def get_team(self, track_id: int) -> int:
        return self._team_labels.get(track_id, TEAM_UNKNOWN)

    def get_team_name(self, track_id: int) -> str:
        return TEAM_NAMES[self.get_team(track_id)]

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        return TEAM_COLORS[self.get_team(track_id)]

    def get_team_rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {TEAM_A: [], TEAM_B: [], TEAM_REF: []}
        for tid, team in self._team_labels.items():
            rosters.setdefault(team, []).append(tid)
        return rosters

    def refine(self) -> None:
        self._fit(label="REFINE")

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """
        Crop the player's torso, pass it through SigLIP, and return the embedding.
        """
        x1 = max(0, int(bbox[0]));  y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1] - 1, int(bbox[2]))
        y2 = min(frame.shape[0] - 1, int(bbox[3]))

        if (x2 - x1) < 16 or (y2 - y1) < 32:
            return None

        crop = frame[y1:y2, x1:x2]
        h_box = crop.shape[0]
        
        t_top = int(h_box * self.torso_ratio[0])
        t_bot = int(h_box * self.torso_ratio[1])
        torso = crop[t_top:t_bot, :]

        if torso.size == 0:
            return None

        # Convert BGR (OpenCV) to RGB (PIL) for SigLIP
        torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(torso_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.siglip_model(**inputs)
            # Use pooler_output (usually a 768-dim vector representing the image)
            embedding = outputs.pooler_output.squeeze().cpu().numpy()

        return embedding

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Aggregate per-player embedding buffers into an (N, 768) feature matrix.
        """
        feats: list[np.ndarray] = []
        tids: list[int] = []

        for tid, obs in self._embed_buffer.items():
            if len(obs) >= self.min_obs:
                # Use the mean embedding for robust representation
                feats.append(np.mean(obs, axis=0))
                tids.append(tid)

        if not feats:
            return np.empty((0, 768), dtype=np.float32), []

        return np.array(feats, dtype=np.float32), tids

    def _fit(self, label: str = "FIT") -> None:
        """
        Apply UMAP for dimensionality reduction, then KMeans for clustering.
        """
        X, tids = self._build_feature_matrix()

        if len(X) < 2:
            print(f"[TeamClusterer] {label} — Not enough valid players ({len(X)}). Waiting...")
            return

        print(f"[TeamClusterer] {label} — Reducing {len(X)} embeddings via UMAP...")
        
        # 1. Dimensionality Reduction
        X_reduced = self.umap_reducer.fit_transform(X)

        # 2. Clustering
        labels = self.kmeans.fit_predict(X_reduced)

        for tid, lab in zip(tids, labels):
            self._team_labels[tid] = int(lab)

        self.is_fitted = True
        n_a = int((labels == 0).sum())
        n_b = int((labels == 1).sum())
        
        print(f"[TeamClusterer] {label} Complete. Team A: {n_a} | Team B: {n_b}")

    def _assign_pending(self) -> None:
        """
        Assign team labels to new tracks using the fitted UMAP and KMeans models.
        """
        for tid, obs in self._embed_buffer.items():
            if tid in self._team_labels:
                continue
            if len(obs) < self.min_obs:
                continue
            
            feat = np.mean(obs, axis=0).reshape(1, -1).astype(np.float32)
            
            # Map high-dimensional embedding to 2D space using fitted UMAP
            feat_reduced = self.umap_reducer.transform(feat)
            
            # Predict team label using fitted KMeans
            label = int(self.kmeans.predict(feat_reduced)[0])
            self._team_labels[tid] = label

    def __repr__(self) -> str:
        return (
            f"TeamClusterer(Fitted={self.is_fitted}, "
            f"Assigned={len(self._team_labels)}, "
            f"Device={self.device})"
        )