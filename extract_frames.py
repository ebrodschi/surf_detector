import cv2
import os

video_path = "new_data/biologia_2025-05-10 08-54-56.mp4"#"biologia.mkv"
output_dir = "biologia_frames2" #"biologia_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Get the video FPS (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 2)  # Every 2 seconds

count = 0
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(filename, frame)
        frame_id += 1

    count += 1

cap.release()
print("Done extracting frames every 2 seconds.")

