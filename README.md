EO/IR Target Recognition & Classification Platform
Framework-Streamlit-red Python-3.9+-blue Model-YOLOv5-blueviolet

A multi-modal Streamlit application for identifying and classifying objects in Electro-Optical (EO) and Infrared (IR) sensor data using the YOLOv5 object detection model.

🎯 Key Features
Multi-Modal Analysis: Process standard videos, live webcam feeds, and batches of thermal images.
Offline Video Processing: Upload a video file (.mp4, .mov, etc.), and the app will:
Extract frames at a set interval.
Run object detection on each frame.
Recompile the frames into a new video with detection boxes drawn on it.
Live Webcam Analysis: Uses your local webcam for real-time object detection, displaying a live video feed and a continuously updated detection log.
Thermal Image Classification: Batch-processes still IR/thermal images from a local directory (flir_images/) and displays the annotated results.
Enhanced Vision: Applies automatic gamma correction to improve visibility in low-light or thermal footage before analysis.
Audible Alerts: A special alarm.mp3 is triggered if the model detects an object classified as "fire."
Comprehensive Logging: All detections are timestamped and logged. The log can be viewed in the app and downloaded as an Excel file (.xlsx) for further analysis.
⚙️ How It Works
The application is split into a user-friendly frontend (app.py) and a powerful processing backend (backend.py).

app.py (Frontend):

Built with Streamlit to create the interactive web interface.
Manages user inputs, file uploads, and application state (e.g., starting/stopping the live feed).
Displays processed videos, images, and data logs in a clean, multi-tab layout.
backend.py (Backend):

Uses Ultralytics YOLOv5 to perform the heavy lifting of object detection. The yolov5s.pt model is loaded once for efficiency.
OpenCV is used for all computer vision tasks: reading video files, extracting/writing frames, and rendering detection boxes.
Pandas is used to structure detection data and export it to Excel.
Handles image enhancement, file I/O, and the alarm system.
🛠️ Tech Stack
Application Framework: Streamlit
Object Detection: PyTorch, Ultralytics YOLOv5
Computer Vision: OpenCV-Python
Data Handling: Pandas, NumPy
Audio: playsound
📂 Project Structure
.

├── 📁 flir_images/
│   └── (Place your thermal .jpg/.png files here)
├── 📁 output/
│   └── (Generated videos and logs will be saved here)
├── 📜 app.py              # The main Streamlit frontend
├── 📜 backend.py            # Core detection and processing functions
├── 🎵 alarm.mp3             # Sound file for fire detection alert
└── 📜 requirements.txt      # Project dependencies
🚀 Getting Started
⚠️ Important Usage Note
For the "Classify Thermal Imagery (IR)" mode to work, you must add your own thermal images (e.g., .jpg, .png) into the flir_images/ directory.

The application reads images directly from this folder.

Prerequisites
Python 3.9 or higher
An active internet connection (for the first run to download the YOLO model)
Installation & Setup
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies from the requirements.txt file:

pip install -r requirements.txt
(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your activated environment.)

Prepare assets:

Place any thermal images you want to analyze into the flir_images/ directory.
Ensure the alarm.mp3 file is present in the root directory for the audio alert feature.
Running the Application
Execute the following command in your terminal:

streamlit run app.py
Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

Use the sidebar to select an analysis mode and follow the on-screen instructions.
