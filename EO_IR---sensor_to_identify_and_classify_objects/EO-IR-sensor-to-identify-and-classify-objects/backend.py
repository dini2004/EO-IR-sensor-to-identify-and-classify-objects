# backend.py

import os
import cv2
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from playsound import playsound
from ultralytics import YOLO

# Load the model once to be used by all functions
try:
    # This will automatically download yolov5s.pt if not present
    model = YOLO("yolov5s.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# === Gamma correction for night vision ===
def adjust_gamma(image, gamma=2.5):
    invGamma = 1.0 / gamma
    table = np.array([(255 * ((i / 255.0) ** invGamma)) for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# === Trigger alarm ===
def trigger_alarm():
    try:
        # Ensure alarm.mp3 is in the same directory or provide a full path
        playsound("alarm.mp3", block=False)
    except Exception as e:
        print(f"Error playing sound: {e}")

# === Offline video processing ===
def extract_frames(video_path, output_folder, interval_sec=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    count = 0
    saved_count = 0
    frame_interval = int(fps * interval_sec)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved_count:04d}.jpg", frame)
            saved_count += 1
        count += 1

    cap.release()
    return saved_count

def detect_objects_on_frames(input_folder, output_folder):
    if not model:
        raise ConnectionError("YOLO model is not loaded.")
    os.makedirs(output_folder, exist_ok=True)
    detection_log = []
    
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
            
        enhanced = adjust_gamma(img, gamma=2.5)
        results = model(enhanced)

        for result in results:
            for i, cls_id in enumerate(result.boxes.cls):
                label = model.names[int(cls_id)]
                confidence = float(result.boxes.conf[i])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detection_log.append({
                    "Frame": img_name, "Object": label,
                    "Confidence": round(confidence, 2), "Timestamp": timestamp
                })
                if label.lower() in ["fire"]:
                    trigger_alarm()
            
            # Save the annotated frame
            annotated_frame = results[0].plot()
            cv2.imwrite(os.path.join(output_folder, img_name), annotated_frame)

    return detection_log

def create_video_from_frames(input_folder, output_path, fps=1):
    """
    Creates a video from a folder of frames.
    """
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not frame_files:
        print("Warning: No frames found in the input folder.")
        return False
    
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Could not read the first frame: {first_frame_path}")
        return False
        
    height, width, _ = first_frame.shape
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use 'mp4v' codec for better .mp4 compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print("Error: Video writer could not be opened. Check codec and permissions.")
        return False

    for file in frame_files:
        frame_path = os.path.join(input_folder, file)
        frame = cv2.imread(frame_path)
        
        # A robust check to ensure the frame was read and has the correct size
        if frame is not None and frame.shape[0] == height and frame.shape[1] == width:
            video.write(frame)
        else:
            print(f"Warning: Skipping frame {file} due to read error or size mismatch.")
    
    print(f"Video saved to {output_path}")
    video.release()
    return True

# === Real-time detection ===
def real_time_detection_generator(stop_event, output_folder="live_detections", interval_sec=1):
    if not model:
        raise ConnectionError("YOLO model is not loaded.")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ConnectionError("Error: Cannot access webcam.")

    frame_count = 0
    saved_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = int(fps * interval_sec)
    detection_log = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = frame.copy() 
        
        if frame_count % frame_interval == 0:
            enhanced = adjust_gamma(frame, gamma=2.5)
            results = model(enhanced)
            annotated_frame = results[0].plot()

            for result in results:
                for i, cls_id in enumerate(result.boxes.cls):
                    label = model.names[int(cls_id)]
                    confidence = float(result.boxes.conf[i])
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detection_log.append({
                        "Frame": f"live_{saved_count:04d}.jpg", "Object": label,
                        "Confidence": round(confidence, 2), "Timestamp": timestamp
                    })
                    if label.lower() in ["fire"]:
                        trigger_alarm()

            img_path = os.path.join(output_folder, f"live_{saved_count:04d}.jpg")
            cv2.imwrite(img_path, annotated_frame)
            saved_count += 1
        
        frame_count += 1
        yield annotated_frame, detection_log

    cap.release()
    cv2.destroyAllWindows()

# === Save log to Excel ===
def save_log_to_excel(detection_log):
    if not detection_log:
        return None
    df = pd.DataFrame(detection_log)
    os.makedirs("output", exist_ok=True)
    excel_path = f"output/detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path