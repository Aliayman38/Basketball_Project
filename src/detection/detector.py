import cv2
import os
from ultralytics import YOLO
from src.team_clustering.clusterer import TeamClusterer
from tqdm import tqdm

class BasketballDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_clusterer = TeamClusterer()

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, imgsz=1280, conf=0.10, verbose=False)[0]
                
                player_bboxes = []
                player_indices = []
                annotated_frame = frame.copy()

                for i, box in enumerate(results.boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 4 and conf > 0.35:
                        player_bboxes.append([x1, y1, x2, y2])
                        player_indices.append(i)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

                    elif cls_id == 0 and conf > 0.10:
                        is_head = False
                        for p_bbox in player_bboxes:
                            px1, py1, px2, py2 = p_bbox
                            head_zone_limit = py1 + (py2 - py1) * 0.15
                            if x1 > px1 and x2 < px2 and y1 < head_zone_limit:
                                is_head = True
                                break
                        
                        if not is_head:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                            cv2.putText(annotated_frame, "Ball", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    elif cls_id == 5 and conf > 0.40:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.putText(annotated_frame, "Ref", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                if player_bboxes:
                    team_labels = self.team_clusterer.assign_teams(frame, player_bboxes)
                    for i, bbox in enumerate(player_bboxes):
                        px1, py1, px2, py2 = bbox
                        team_color = (0, 255, 0) if team_labels[i] == 0 else (255, 0, 0)
                        cv2.putText(annotated_frame, f"Team {team_labels[i]}", (px1, py1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, team_color, 2)

                out.write(annotated_frame)
                pbar.update(1)
        
        cap.release()
        out.release()