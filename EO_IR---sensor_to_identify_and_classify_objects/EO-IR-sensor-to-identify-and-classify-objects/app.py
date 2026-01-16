# app.py

import streamlit as st
import cv2
import os
import time
from threading import Event
import pandas as pd
from backend import (
    extract_frames,
    detect_objects_on_frames,
    create_video_from_frames,
    real_time_detection_generator,
    save_log_to_excel
)

# --- Page Configuration ---
st.set_page_config(
    page_title="EO/IR Target Recognition",
    page_icon="🎯",
    layout="wide"
)

# --- Initialize Session State ---
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = Event()
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'run_realtime' not in st.session_state:
    st.session_state.run_realtime = False


# --- Main UI ---
st.title("🎯 EO/IR Target Identification & Classification")
st.markdown("A platform for analyzing Electro-Optical (EO) and Infrared (IR) sensor data to identify and classify objects using YOLOv5.")

# --- Sidebar Control Panel ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.info(
    """
    **EO:** Electro-Optical (Visual Light)\n
    **IR:** Infrared (Thermal)
    """
)

# --- Added a new, simpler mode for viewing files ---
mode = st.sidebar.radio(
    "Select Analysis Mode:",
    (
        "Analyze Video File (EO)", 
        "Live EO Stream Analysis", 
        "Classify Thermal Imagery (IR)",
        "View Local Video File" #<-- NEW MODE
    )
)


# --- Mode 1: Analyze Video File (EO) ---
if mode == "Analyze Video File (EO)":
    st.header("🎥 Analyze Video File (Electro-Optical)")

    uploaded_file = st.file_uploader(
        "Upload a video file for analysis...", 
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(video_path)
        
        if st.button("Start Processing", key="start_offline", use_container_width=True):
            st.session_state.detection_log = []
            
            with st.spinner("[1/3] Extracting frames from video..."):
                frames_extracted = extract_frames(video_path, "offline_frames")
                st.success(f"Step 1 Complete: Extracted {frames_extracted} frames.")

            with st.spinner("[2/3] Performing object detection... (This may take a while)"):
                detection_log = detect_objects_on_frames("offline_frames", "offline_detections")
                st.session_state.detection_log.extend(detection_log)
                st.success("Step 2 Complete: Object detection finished.")

            with st.spinner("[3/3] Recompiling video with annotations..."):
                output_video_path = "output/offline_result.mp4"
                video_created = create_video_from_frames("offline_detections", output_video_path)
                if video_created:
                    st.success(f"Step 3 Complete: Processed video saved to {output_video_path}.")
                else:
                    st.error("Step 3 Failed: Could not create the output video.")

            if video_created:
                st.balloons()
                st.subheader("📊 Analysis Results")

                with st.expander("🎬 Show Processed Video", expanded=True):
                    st.video(output_video_path)
                    
                    try:
                        with open(output_video_path, "rb") as video_file:
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_file,
                                file_name=os.path.basename(output_video_path),
                                mime="video/mp4"
                            )
                    except FileNotFoundError:
                        st.error("Could not find the processed video file to create a download link.")
                
                log_path = save_log_to_excel(st.session_state.detection_log)
                if log_path:
                    with st.expander("📄 Show Detection Log"):
                        df = pd.DataFrame(st.session_state.detection_log)
                        st.dataframe(df)
                        with open(log_path, "rb") as f:
                            st.download_button(
                                "📥 Download Log (Excel)", 
                                f, 
                                file_name=os.path.basename(log_path)
                            )
    else:
        st.info("Please upload a video file to begin analysis.")


# --- Mode 2: Live EO Stream Analysis ---
elif mode == "Live EO Stream Analysis":
    st.header("🛰️ Live EO Stream Analysis")
    st.info("Engages the local webcam for real-time object detection.")

    col1, col2 = st.columns(2)
    start_button = col1.button("▶ Start Live Feed", key="start_realtime", use_container_width=True)
    stop_button = col2.button("⏹️ Stop Live Feed", key="stop_realtime", use_container_width=True)

    if start_button:
        st.session_state.stop_event.clear()
        st.session_state.run_realtime = True
        st.rerun()
    
    if stop_button:
        st.session_state.stop_event.set()
        st.session_state.run_realtime = False
        st.info("Live feed stopped.")
        time.sleep(1) 
        st.rerun()

    if st.session_state.get("run_realtime", False):
        frame_placeholder = st.empty()
        log_placeholder = st.empty()
        final_log = []

        try:
            live_generator = real_time_detection_generator(st.session_state.stop_event)
            for annotated_frame, detection_log in live_generator:
                final_log = detection_log
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption="Live EO Feed", channels="RGB", use_column_width=True)
                
                if detection_log:
                    with log_placeholder.container():
                        st.markdown("##### Detection Log (Live)")
                        df = pd.DataFrame(detection_log).sort_values(by="Timestamp", ascending=False)
                        st.dataframe(df, height=200, use_container_width=True)
            
            log_path = save_log_to_excel(final_log)
            if log_path:
                st.success(f"✅ Session Ended. Final log saved to {log_path}.")
                with open(log_path, "rb") as f:
                    st.download_button("📥 Download Final Log", f, file_name=os.path.basename(log_path))

        except ConnectionError as e:
            st.error(str(e))
            st.session_state.run_realtime = False
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.run_realtime = False


# --- Mode 3: Classify Thermal Imagery (IR) ---
elif mode == "Classify Thermal Imagery (IR)":
    st.header("🔥 Classify Thermal Imagery (Infrared)")
    flir_folder = "flir_images"
    
    if not os.path.exists(flir_folder) or not os.listdir(flir_folder):
        st.warning(f"Input folder is empty or not found! Please add thermal images to the `{flir_folder}` directory.")
    else:
        st.info(f"Ready to process images from the `{flir_folder}` directory.")

        if st.button("Start IR Analysis", key="start_flir", use_container_width=True):
            st.session_state.detection_log = []
            output_folder = "flir_detections"
            
            with st.spinner("Analyzing thermal images..."):
                detection_log = detect_objects_on_frames(flir_folder, output_folder)
                st.session_state.detection_log.extend(detection_log)
                st.success(f"Analysis Complete. Annotated images saved to `{output_folder}`.")
            
            st.balloons()
            st.subheader("📊 Analysis Results")
            
            log_path = save_log_to_excel(st.session_state.detection_log)
            if log_path:
                with st.expander("📄 Show Detection Log"):
                    st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
                    with open(log_path, "rb") as f:
                        st.download_button("📥 Download Log (Excel)", f, file_name=os.path.basename(log_path))

            with st.expander("🖼️ Show Processed Images", expanded=True):
                processed_images = sorted([f for f in os.listdir(output_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
                if processed_images:
                    st.image(
                        [os.path.join(output_folder, img) for img in processed_images],
                        caption=[img for img in processed_images],
                        width=200 
                    )
                else:
                    st.warning("No processed images to display.")

# --- NEW MODE: View a local video file directly ---
elif mode == "View Local Video File":
    st.header("🎬 View Local Video File")
    st.info("This mode allows you to directly upload and view a video file from your computer, skipping the processing steps.")
    
    local_video_file = st.file_uploader(
        "Upload a video file to view", 
        type=["mp4", "mov", "avi", "mkv"]
    )
    
    if local_video_file is not None:
        st.video(local_video_file)
        
        st.download_button(
            label="📥 Download This Video",
            data=local_video_file,
            file_name=local_video_file.name,
            mime=local_video_file.type
        )