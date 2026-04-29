from ultralytics import YOLO
import cv2
from src.team_clustering.clusterer import TeamClusterer

class BasketballDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_clusterer = TeamClusterer()

    def predict_and_track_teams(self, frame):
        results = self.model.predict(source=frame, imgsz=1280, conf=0.25)[0]
        
        player_bboxes = []
        player_indices = []

        for i, box in enumerate(results.boxes):
            if int(box.cls) == 4:
                player_bboxes.append(box.xyxy[0].tolist())
                player_indices.append(i)

        if player_bboxes:
            team_labels = self.team_clusterer.assign_teams(frame, player_bboxes)
            return results, player_indices, team_labels
        
        return results, [], []

    def save_prediction(self, results, save_path):
        results.save(filename=save_path)