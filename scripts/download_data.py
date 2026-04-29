from roboflow import Roboflow
import os

save_dir = "data/dataset"
os.makedirs(save_dir, exist_ok=True)

# Your confirmed API Key
rf = Roboflow(api_key="cb2yWxuN5eiV3gOHh2gR")
project = rf.workspace("workspace-5ujvu").project("basketball-players-fy4c2-vfsuv")
version = project.version(11)
dataset = project.version(11).download("yolov8", location="data/dataset")
print("SUCCESS: Dataset version 11 is now on your machine!")
