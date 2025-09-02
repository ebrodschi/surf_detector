from ultralytics import YOLO

import os

# Define the base path where your dataset is located
base_path = os.path.abspath("data")  # Assumes your "data" folder is in the current directory

# Build absolute paths for train, validation, and test
data_yaml_path = os.path.join(base_path, "data.yaml")
print(data_yaml_path)
train_path = os.path.join(base_path, "train", "images")
test_path = os.path.join(base_path, "test", "images")

# Load a YOLOv8 model (you can also use 'yolov8n', 'yolov8m', 'yolov8l')
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(
    data=data_yaml_path,      # Path to your Roboflow-generated YAML
    epochs=50,
    imgsz=640,
    batch=8,
    name="surfer_detector_v2", # Folder for saved weights
    project="runs/train",   # Where results will be saved
    val=True                # Enable validation
)

# Evaluate on test set if needed
metrics = model.val(data=data_yaml_path, split="test")  # optional
