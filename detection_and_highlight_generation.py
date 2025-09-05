
import time
import argparse

import os, sys, subprocess

def ensure_pkgs():
    try:
        import moviepy  # noqa
    except ModuleNotFoundError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "moviepy>=1.0.3", "imageio-ffmpeg>=0.4.9"
        ])

ensure_pkgs()

# Ensure ffmpeg binary is available to MoviePy
from imageio_ffmpeg import get_ffmpeg_exe
os.environ["IMAGEIO_FFMPEG_EXE"] = get_ffmpeg_exe()


from moviepy.editor import VideoFileClip

def transcode_to_h264_web(mp4_in: str, mp4_out: str, fps: int = None):
    """
    Creates a browser-compatible MP4: H.264 (yuv420p) + faststart.
    Your OpenCV output has no audio, so we disable audio (-an equivalent).
    """
    clip = VideoFileClip(mp4_in, audio=False)
    clip.write_videofile(
        mp4_out,
        codec="libx264",
        audio=False,
        fps=fps or clip.fps or 30,
        preset="veryfast",
        ffmpeg_params=["-movflags", "faststart", "-pix_fmt", "yuv420p"],
        threads=0,  # auto
        logger=None,
    )
    clip.close()

def detect_and_highlight(input_video_path, output_video_path):

    import cv2
    import numpy as np
    from ultralytics import YOLO

    # Input and output paths
    model = YOLO("runs/train/surfer_detector_v2/weights/best.pt")

    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_second = int(fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, frames_per_second, (width, height))

    frame_count = 0
    results = None

    SURFER_RIDING_IDX = 1  # from data.yaml

    # Detection persistence parameters
    INITIAL_CONFIRMATION_SECONDS = 3
    END_GRACE_PERIOD_SECONDS = 3
    INITIAL_CONFIRMATION_FRAMES = INITIAL_CONFIRMATION_SECONDS * frames_per_second
    END_GRACE_PERIOD_FRAMES = END_GRACE_PERIOD_SECONDS * frames_per_second

    # Spatial tracking parameters
    PROXIMITY_THRESHOLD = 100
    MAX_SURFER_ID = 0

    # Timing variables
    total_surf_time_seconds = 0
    total_waves_surfed = 0
    active_surfers = {}

    # List to store surf segments for video generation
    surf_segments = []  # [(start_frame, end_frame, surfer_id), ...]


    def calculate_box_center(box):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return ((x1 + x2) / 2, (y1 + y2) / 2)


    def find_closest_surfer(new_center, active_surfers, threshold):
        """Find the closest active surfer within the proximity threshold"""
        min_distance = float("inf")
        closest_id = None

        for surfer_id, surfer_data in active_surfers.items():
            if "last_position" in surfer_data:
                last_pos = surfer_data["last_position"]
                distance = np.sqrt(
                    (new_center[0] - last_pos[0]) ** 2 + (new_center[1] - last_pos[1]) ** 2
                )

                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_id = surfer_id

        return closest_id


    print("Processing video and detecting surf segments...")

    # First pass: Detect all surf segments
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection every 3rd frame to save processing
        if frame_count % 3 == 0:
            results = model(frame, imgsz=320)

        # Process detections and assign to surfers
        current_detections = []
        if results:
            boxes = results[0].boxes
            if len(boxes) > 0:
                mask = boxes.cls == SURFER_RIDING_IDX
                surfer_boxes = boxes[mask]

                # Process each detected surfer
                for box in surfer_boxes:
                    center = calculate_box_center(box)
                    closest_id = find_closest_surfer(
                        center, active_surfers, PROXIMITY_THRESHOLD
                    )

                    if closest_id is not None:
                        # Update existing surfer
                        active_surfers[closest_id]["last_position"] = center
                        active_surfers[closest_id]["detection_history"].append(True)
                        active_surfers[closest_id]["frames_since_last"] = 0
                        active_surfers[closest_id]["last_detection_frame"] = frame_count
                        current_detections.append((closest_id, box))
                    else:
                        # Create new surfer
                        MAX_SURFER_ID += 1
                        new_id = MAX_SURFER_ID
                        active_surfers[new_id] = {
                            "last_position": center,
                            "detection_history": [True],
                            "confirmed": False,
                            "frames_since_last": 0,
                            "surf_start_frame": None,
                            "last_detection_frame": frame_count,
                            "total_surf_time": 0,
                        }
                        current_detections.append((new_id, box))

        # Update all active surfers
        surfers_to_remove = []
        for surfer_id, surfer_data in active_surfers.items():
            detected_this_frame = any(det[0] == surfer_id for det in current_detections)

            if not detected_this_frame:
                surfer_data["detection_history"].append(False)
                surfer_data["frames_since_last"] += 1

            # Limit detection history size
            if len(surfer_data["detection_history"]) > INITIAL_CONFIRMATION_FRAMES:
                surfer_data["detection_history"].pop(0)

            # Check for initial confirmation
            if (
                not surfer_data["confirmed"]
                and len(surfer_data["detection_history"]) >= INITIAL_CONFIRMATION_FRAMES
            ):
                recent_detections = sum(
                    surfer_data["detection_history"][-INITIAL_CONFIRMATION_FRAMES:]
                )
                detection_ratio = recent_detections / INITIAL_CONFIRMATION_FRAMES

                if detection_ratio >= 0.7:
                    surfer_data["confirmed"] = True
                    surfer_data["surf_start_frame"] = frame_count
                    print(
                        f"✓ Surfer {surfer_id} confirmed at {frame_count/frames_per_second:.1f}s"
                    )

            # Check if surfer should be removed (exceeded grace period)
            if surfer_data["frames_since_last"] > END_GRACE_PERIOD_FRAMES:
                if surfer_data["confirmed"]:
                    # Record surf segment
                    start_frame = surfer_data["surf_start_frame"]
                    end_frame = surfer_data["last_detection_frame"]
                    surf_segments.append((start_frame, end_frame, surfer_id))

                    surf_duration_seconds = (end_frame - start_frame) / frames_per_second
                    total_surf_time_seconds += surf_duration_seconds
                    total_waves_surfed += 1
                    print(
                        f"✓ Surf segment recorded: Surfer {surfer_id}, {surf_duration_seconds:.1f}s"
                    )
                surfers_to_remove.append(surfer_id)

        # Remove inactive surfers
        for surfer_id in surfers_to_remove:
            del active_surfers[surfer_id]

        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames...")

    # Handle any remaining active surfers at end of video
    for surfer_id, surfer_data in active_surfers.items():
        if surfer_data["confirmed"]:
            start_frame = surfer_data["surf_start_frame"]
            end_frame = surfer_data["last_detection_frame"]
            surf_segments.append((start_frame, end_frame, surfer_id))
            surf_duration_seconds = (end_frame - start_frame) / frames_per_second
            total_surf_time_seconds += surf_duration_seconds
            total_waves_surfed += 1
            print(f"✓ Final surf segment: Surfer {surfer_id}, {surf_duration_seconds:.1f}s")

    print(f"\n=== DETECTION COMPLETE ===")
    print(f"Found {len(surf_segments)} surf segments")
    print(f"Total surf time: {total_surf_time_seconds:.1f} seconds")
    print(f"Total waves: {total_waves_surfed}")

    # Second pass: Generate highlight video
    print("\nGenerating highlight video...")
    cap.release()
    cap = cv2.VideoCapture(input_video_path)

    # Sort surf segments by start frame
    surf_segments.sort(key=lambda x: x[0])

    # Add buffer frames before and after each segment
    BUFFER_SECONDS = 2  # Add 2 seconds before and after each surf segment
    BUFFER_FRAMES = BUFFER_SECONDS * frames_per_second

    # Merge overlapping segments
    merged_segments = []
    for start, end, surfer_id in surf_segments:
        buffered_start = max(0, start - BUFFER_FRAMES)
        buffered_end = end + BUFFER_FRAMES

        # Check if this overlaps with the last segment
        if merged_segments and buffered_start <= merged_segments[-1][1]:
            # Merge with previous segment
            merged_segments[-1] = (
                merged_segments[-1][0],
                max(merged_segments[-1][1], buffered_end),
                f"{merged_segments[-1][2]},{surfer_id}",
            )
        else:
            merged_segments.append((buffered_start, buffered_end, str(surfer_id)))

    print(f"Merged into {len(merged_segments)} segments")

    # Write segments to output video
    current_frame = 0
    segment_idx = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if current frame is in any segment
        if segment_idx < len(merged_segments):
            start, end, surfer_ids = merged_segments[segment_idx]

            if current_frame >= start and current_frame <= end:
                # Add timestamp and segment info
                timestamp = current_frame / frames_per_second
                cv2.putText(
                    frame,
                    f"Time: {timestamp:.1f}s | Surfer(s): {surfer_ids}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                out.write(frame)
                frames_written += 1
            elif current_frame > end:
                segment_idx += 1

        current_frame += 1

        if current_frame % 1000 == 0:
            print(f"Processed {current_frame} frames for output...")

    cap.release()
    out.release()
    
    h264_path = output_video_path.replace(".mp4", "_h264.mp4")
    transcode_to_h264_web(output_video_path, h264_path, fps=frames_per_second)
    print(f"Transcoded for web playback: {h264_path}")


    highlight_duration = frames_written / frames_per_second
    compression_ratio = (highlight_duration / (current_frame / frames_per_second)) * 100

    print(f"\n=== VIDEO GENERATION COMPLETE ===")
    print(f"Output saved to: {output_video_path}")
    print(f"Original video: {current_frame / frames_per_second:.1f} seconds")
    print(f"Highlight video: {highlight_duration:.1f} seconds")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print(f"Frames written: {frames_written}")

if __name__ == "__main__":
    
    # set parset arguments
    
    parser = argparse.ArgumentParser(description="Detect and highlight surf segments in a video.")
    parser.add_argument("--input_video_path", type=str, required=False, help="Path to the input video file.")
    parser.add_argument("--output_video_path", type=str, required=False, help="Path to save the output highlight video.")
    args = parser.parse_args()
    
    input_video_path = args.input_video_path if args.input_video_path else "biologia_para_demo.mp4" #new_data/biologia_2025-05-10 08-54-56.mp4
    timestamp = time.strftime("%Y%m%d-%H%M%S")  
    output_video_path = args.output_video_path if args.output_video_path else f"surf_highlights_{timestamp}.mp4"
    print(f"Output video path set to: {output_video_path}")
    
    detect_and_highlight(input_video_path, output_video_path)