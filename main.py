import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


video_path = "data/test_3.mp4"     # 1. هون بتطلب الفيديو (المدخل)
output_path = "runs/final_output.mp4"  # 2. وهون بتطلع "الحياة" (المخرج)
CLASS_BALL = 0
CLASS_CLOCK = 1
CLASS_HOOP = 2
CLASS_OVERLAY = 3
CLASS_PLAYER = 4
CLASS_REF = 5

TEAM_A = 0
TEAM_B = 1
TEAM_REF = 2
TEAM_UNKNOWN = -1

TEAM_COLORS = {
    TEAM_A: (235, 110, 40),
    TEAM_B: (40, 200, 60),
    TEAM_REF: (50, 50, 220),
    TEAM_UNKNOWN: (160, 160, 160),
}

TEAM_NAMES = {
    TEAM_A: "Team A",
    TEAM_B: "Team B",
    TEAM_REF: "Referee",
    TEAM_UNKNOWN: "Unknown",
}

class TeamClusterer:
    def __init__(self, warm_up_frames=60, torso_ratio=(0.12, 0.52), method="kmeans", min_color_obs=5):
        self.warm_up_frames = warm_up_frames
        self.torso_ratio = torso_ratio
        self.method = method
        self.min_color_obs = min_color_obs
        self._color_buffer = defaultdict(list)
        self._team_labels = {}
        self._model = None
        self.is_fitted = False
        self._frame_idx = 0

    def update(self, frame, tracked_dets):
        for det in tracked_dets:
            cid = int(det["class_id"])
            tid = int(det["track_id"])
            
            if cid == CLASS_REF:
                self._team_labels[tid] = TEAM_REF
                continue
                
            if cid != CLASS_PLAYER:
                continue
                
            colour = self._extract_jersey_hsv(frame, det["bbox"])
            if colour is not None:
                self._color_buffer[tid].append(colour)
                
        self._frame_idx += 1
        
        if self._frame_idx == self.warm_up_frames and not self.is_fitted:
            self._fit()
            
        if self.is_fitted:
            self._assign_pending()

    def get_team(self, track_id):
        return self._team_labels.get(track_id, TEAM_UNKNOWN)

    def get_team_name(self, track_id):
        return TEAM_NAMES[self.get_team(track_id)]

    def get_color(self, track_id):
        return TEAM_COLORS[self.get_team(track_id)]

    def get_team_rosters(self):
        rosters = {TEAM_A: [], TEAM_B: [], TEAM_REF: []}
        for tid, team in self._team_labels.items():
            rosters.setdefault(team, []).append(tid)
        return rosters

    def refine(self):
        self._fit(label="REFINE")

    def print_roster(self):
        rosters = self.get_team_rosters()
        for team_id in (TEAM_A, TEAM_B, TEAM_REF):
            tids = sorted(rosters.get(team_id, []))
            name = TEAM_NAMES[team_id]

    def _extract_jersey_hsv(self, frame, bbox):
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1] - 1, int(bbox[2]))
        y2 = min(frame.shape[0] - 1, int(bbox[3]))

        if (x2 - x1) < 8 or (y2 - y1) < 16:
            return None

        crop = frame[y1:y2, x1:x2]
        h_box = crop.shape[0]
        t_top = int(h_box * self.torso_ratio[0])
        t_bot = int(h_box * self.torso_ratio[1])
        torso = crop[t_top:t_bot, :]

        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        mask = (pixels[:, 1] > 40) & (pixels[:, 2] > 30)
        valid = pixels[mask]

        return np.median(valid if len(valid) >= 10 else pixels, axis=0)

    def _build_feature_matrix(self):
        feats = []
        tids = []

        for tid, obs in self._color_buffer.items():
            if len(obs) >= self.min_color_obs:
                feats.append(np.median(obs, axis=0))
                tids.append(tid)

        if not feats:
            return np.empty((0, 3), dtype=np.float32), []

        return np.array(feats, dtype=np.float32), tids

    def _fit(self, label="FIT"):
        X, tids = self._build_feature_matrix()

        if len(X) < 2:
            return

        if self.method == "gmm":
            self._model = GaussianMixture(n_components=2, random_state=42, n_init=5, covariance_type="full")
            labels = self._model.fit_predict(X)
        else:
            self._model = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = self._model.fit_predict(X)

        for tid, lab in zip(tids, labels):
            self._team_labels[tid] = int(lab)

        self.is_fitted = True

    def _assign_pending(self):
        for tid, obs in self._color_buffer.items():
            if tid in self._team_labels:
                continue
            if len(obs) < self.min_color_obs:
                continue
            feat = np.median(obs, axis=0).reshape(1, -1).astype(np.float32)
            label = int(self._model.predict(feat)[0])
            self._team_labels[tid] = label