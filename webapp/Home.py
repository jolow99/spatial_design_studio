import streamlit as st
import pyvista as pv
import subprocess

def is_xvfb_running():
    try:
        result = subprocess.run(["ps", "-e"], capture_output=True, text=True)
        return "Xvfb" in result.stdout
    except FileNotFoundError:
        print("`ps` command not available, skipping check.")
        return False

if not is_xvfb_running():
    print("Xvfb is not running, attempting to start...")
    pv.start_xvfb()
    print("Xvfb started successfully.")
else:
    print("Xvfb is already running.")
    
pages = {
    "Waves 🌊": [
        st.Page("1_Introduction.py", title="Introduction"),
    ],
    "Our Process": [
        st.Page("2_Experiment_Design.py", title="Experiment Design"),
        st.Page("3_Data_Visualization.py", title="Data Visualization"),
        st.Page("4_3D_Model_Gallery.py", title="3D Model Gallery"),
    ],
        "Try it Yourself": [
        st.Page("5_Convert_Mesh_to_Point_Cloud.py", title="1. Convert Mesh to Point Cloud"),
        st.Page("6_Run_Predictions.py", title="2. Run Predictions"),
    ],
}

pg = st.navigation(pages)
pg.run()