import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClusterer:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.is_fitted = False

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        
        height, width, _ = player_img.shape
        
        shirt_img = player_img[int(height*0.2):int(height*0.5), int(width*0.2):int(width*0.8)]
        
        if shirt_img.size == 0:
            return np.array([0, 0, 0])

        hsv_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv_img, axis=(0, 1))
        return avg_color

    def assign_teams(self, frame, player_bboxes):
        player_colors = []
        for bbox in player_bboxes:
            color = self.get_player_color(frame, bbox)
            player_colors.append(color)
        
        player_colors = np.array(player_colors)
        
        if not self.is_fitted:
            self.kmeans.fit(player_colors)
            self.is_fitted = True
            return self.kmeans.labels_.tolist()
        
        labels = self.kmeans.predict(player_colors)
        return labels.tolist()