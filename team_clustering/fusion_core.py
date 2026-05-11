import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class TeamAssignmentTracker:
    def __init__(self, lock_after=10):
        self.history = {}
        self.locked = {}
        self.lock_after = lock_after

    def update(self, player_id, new_team):
        if player_id in self.locked:
            return self.locked[player_id]
            
        if player_id not in self.history:
            self.history[player_id] = []
            
        if new_team is not None:
            self.history[player_id].append(new_team)
            
        if not self.history[player_id]:
            return None
            
        recent = self.history[player_id][-15:]
        team = max(set(recent), key=recent.count)
        
        if len(self.history[player_id]) >= self.lock_after:
            self.locked[player_id] = team
            
        return team


class MultiSignalClusterer:
    def __init__(self, team_0_desc, team_1_desc):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_prompts = [team_0_desc, team_1_desc]
        self.team_names = {0: 'T1', 1: 'T2'}

    def extract_jersey_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        
        torso_y1 = int(y1 + h * 0.40)
        torso_y2 = int(y1 + h * 0.75)
        margin = int((x2 - x1) * 0.1)
        
        fh, fw = frame.shape[:2]
        cy1, cy2 = max(0, torso_y1), min(fh, torso_y2)
        cx1, cx2 = max(0, x1 + margin), min(fw, x2 - margin)
        
        if cy2 <= cy1 or cx2 <= cx1:
            return None
            
        return frame[cy1:cy2, cx1:cx2]

    def get_dominant_color_hsv(self, crop):
        if crop is None or crop.size == 0:
            return None
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
        pixels = hsv[mask > 0].reshape(-1, 3)
        
        if len(pixels) < 50:
            return None
            
        kmeans = KMeans(n_clusters=1, init='k-means++', n_init=3)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[0]

    def cluster_by_reid(self, reid_model, player_crops):
        embeddings = []
        valid_indices = []
        
        for idx, crop in enumerate(player_crops):
            if crop is not None and crop.size > 0:
                feat = reid_model.extract(crop)
                embeddings.append(feat)
                valid_indices.append(idx)
                
        if len(embeddings) < 2:
            return None
            
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels, valid_indices

    def assign_team_clip(self, crop, threshold=0.6):
        if crop is None or crop.size == 0:
            return None
            
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        inputs = self.clip_processor(
            text=self.text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            
        max_prob = torch.max(probs).item()
        if max_prob < threshold:
            return None
            
        return int(probs.argmax().item())