import cv2
import os
from ultralytics import YOLO
from src.team_clustering.clusterer import TeamClusterer

class BasketballDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_clusterer = TeamClusterer()

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎬 Processing Video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Detection
            results = self.model.predict(frame, imgsz=1280, conf=0.25, verbose=False)[0]
            
            player_bboxes = []
            player_indices = []

            for i, box in enumerate(results.boxes):
                if int(box.cls) == 4: 
                    player_bboxes.append(box.xyxy[0].tolist())
                    player_indices.append(i)

            # Team Clustering
            annotated_frame = results.plot()
            if player_bboxes:
                team_labels = self.team_clusterer.assign_teams(frame, player_bboxes)
                for i, idx in enumerate(player_indices):
                    x1, y1, x2, y2 = map(int, results.boxes[idx].xyxy[0])
                    team_color = (0, 255, 0) if team_labels[i] == 0 else (255, 0, 0)
                    cv2.putText(annotated_frame, f"Team {team_labels[i]}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, team_color, 2)

            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"✅ Processing Complete! Saved to: {output_path}")