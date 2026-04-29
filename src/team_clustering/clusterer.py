import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClusterer:
    def __init__(self):
        self.team_colors = {}

    def get_player_color(self, frame, bbox):
     x1, y1, x2, y2 = map(int, bbox)
     player_img = frame[y1:y2, x1:x2]
    
     height, width, _ = player_img.shape
     shirt_img = player_img[int(height*0.2):int(height*0.5), int(width*0.2):int(width*0.8)]
    
     if shirt_img.size == 0: return np.array([0,0,0])
 
     hsv_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
     
     avg_color = np.mean(hsv_img, axis=(0, 1))
     return avg_color

    def assign_teams(self, frame, player_bboxes):
        player_colors = []
        for bbox in player_bboxes:
            player_colors.append(self.get_player_color(frame, bbox))
        
        if len(player_colors) < 2: return [0] * len(player_colors)

        kmeans = KMeans(n_clusters=2, n_init=10)
        teams = kmeans.fit_predict(player_colors)
        return teams