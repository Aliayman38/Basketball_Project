"""
team_clustering/clusterer.py
─────────────────────────────
Zero-Shot Team Clustering using OpenAI CLIP.
No collect/fit phase needed — predicts directly from jersey crop.
"""


import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPTeamClusterer:
    TEAM_COLORS = {
        0: (255, 255, 255),   # Team 1
        1: (0,   0,   255),   # Team 2
    }
    TEAM_NAMES = {0: 'T1', 1: 'T2'}

    def __init__(
        self,
        team_0_desc: str = "a basketball player wearing a white jersey",
        team_1_desc: str = "a basketball player wearing a dark blue jersey",
    ):
        print("\n⏳ Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_prompts = [team_0_desc, team_1_desc]
        print(f"✅ CLIP ready on [{self.device}]")
        print(f"   Team 1: {team_0_desc}")
        print(f"   Team 2: {team_1_desc}")

    def predict(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, threshold: float = 0.80) -> int | None:
        """Predict team from a live frame crop. Returns 0, 1, or None if uncertain."""
        h, w = y2 - y1, x2 - x1
        
        # 1. اقتطاع منطقة الجذع فقط (40% إلى 80% من الارتفاع)
        cy1 = int(y1 + h * 0.40)
        cy2 = int(y1 + h * 0.80)
        cx1 = int(x1 + w * 0.15)
        cx2 = int(x2 - w * 0.15)

        fh, fw = frame.shape[:2]
        cy1, cy2 = max(0, cy1), min(fh, cy2)
        cx1, cx2 = max(0, cx1), min(fw, cx2)

        if cy2 <= cy1 or cx2 <= cx1:
            return None

        bgr_crop = frame[cy1:cy2, cx1:cx2]
        if bgr_crop.size == 0:
            return None

        rgb_crop  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        inputs = self.processor(
            text=self.text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = outputs.logits_per_image.softmax(dim=1)[0] # استخراج المصفوفة 1D

        # 2. تطبيق عتبة الثقة (Threshold)
        max_prob = torch.max(probs).item()
        if max_prob < threshold:
            return None  # رفض التصنيف إذا كانت الثقة ضعيفة

        return int(probs.argmax().item())