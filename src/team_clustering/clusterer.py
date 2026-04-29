import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClusterer:
    def __init__(self):
        self.team_colors = {}

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]
        
        shirt_zone = player_img[0:int(player_img.shape[0]*0.4), :]
        
        if shirt_zone.size == 0: return np.array([0,0,0])

        pixels = shirt_zone.reshape(-1, 3)
        kmeans = KMeans(n_clusters=1, n_init=1)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[0]

    def assign_teams(self, frame, player_bboxes):
        player_colors = []
        for bbox in player_bboxes:
            player_colors.append(self.get_player_color(frame, bbox))
        
        if len(player_colors) < 2: return [0] * len(player_colors)

        kmeans = KMeans(n_clusters=2, n_init=10)
        teams = kmeans.fit_predict(player_colors)
        return teams