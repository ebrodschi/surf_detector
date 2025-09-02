import streamlit as st
import cv2
import numpy as np
import os
from detection_and_highlight_generation import detect_and_highlight 
from texts import texts


# --- Language selector ---
lang = st.selectbox("Choose language / Elegí idioma", ["Español", "English"])

# --- UI ---
st.title(texts["title"][lang])

st.header(texts["algo_header"][lang])
st.write(texts["algo_desc"][lang])

st.header(texts["training_header"][lang])
st.write(texts["training_desc"][lang])

st.header(texts["demo_header"][lang])
demo_video_path = "surf_detector_demo.mp4"
if os.path.exists(demo_video_path):
    st.video(demo_video_path)
else:
    st.error(texts["demo_not_found"][lang])

st.header(texts["upload_header"][lang])
uploaded_file = st.file_uploader(texts["upload_prompt"][lang], type=["mp4", "mov", "avi"])

temp_video_path = "temp_video.mp4"
output_video_path = "output_video.mp4"

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write(texts["running_detection"][lang])
    # Call the detection function (assuming it takes input and output paths)
    detect_and_highlight(temp_video_path, output_video_path)
    
    import time
    time.sleep(5)  # Small delay to ensure file is ready

    # Display the output video
    if os.path.exists(output_video_path):
        st.video(output_video_path)
        # Add button to delete temp files
        if st.button("Borrar archivos temporales / Delete temporary files"):
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
                st.success("Archivos temporales borrados / Temporary files deleted.")
            else:
                st.warning("No se encontraron archivos temporales para borrar.")
    else:
        st.error(texts["error_processing"][lang])

