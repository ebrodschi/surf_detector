import streamlit as st
import time
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
demo_video_path = "surf_detector_demo.mp4" #surf_detector_demo
if os.path.exists(demo_video_path):
    st.video(demo_video_path)
else:
    st.error(texts["demo_not_found"][lang])

st.header(texts["upload_header"][lang])
uploaded_file = st.file_uploader(texts["upload_prompt"][lang], type=["mp4", "mov", "avi"])

temp_video_path = "temp_video.mp4"
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
output_video_path = f"output_video_{timestamp}.mp4"

if uploaded_file is not None:
    if st.button("Run detection"):
        # Save the uploaded video to a temporary file
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write(texts["running_detection"][lang])
        # Call the detection function (assuming it takes input and output paths)
        detect_and_highlight(temp_video_path, output_video_path)
        
        import time
        time.sleep(5)  # Small delay to ensure file is ready
        st.session_state["detection_done"] = True
    


# Display the output video
if st.session_state.get("detection_done", False) and os.path.exists(output_video_path):
    h264_path = output_video_path.replace(".mp4", "_h264.mp4")
    st.video(h264_path)
    # Add button to delete temp files
    if st.button("Borrar archivos temporales / Delete temporary files"):
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        if os.path.exists(h264_path):
            os.remove(h264_path)
            st.success("Archivos temporales borrados / Temporary files deleted.")
            st.session_state["detection_done"] = False        
        else:
            st.warning("No se encontraron archivos temporales para borrar.")


