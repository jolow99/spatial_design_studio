import os
import re
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

CURVILINEAR_DIR = "assets/models/curvilinear"
RECTILINEAR_DIR = "assets/models/rectilinear"

# Utility: Extract numbers from filenames for sorting
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_model(file_path):
    if file_path.endswith(".stl"):
        return pv.STLReader(file_path).read()
    elif file_path.endswith(".obj"):
        return pv.OBJReader(file_path).read()
    else:
        return None

def render_model(file_path):
    mesh = load_model(file_path)
    if mesh is None:
        return None

    plotter = pv.Plotter(border=False, window_size=[400, 400])
    plotter.background_color = "#f0f8ff"
    plotter.add_mesh(mesh, color="orange", specular=0.5)
    plotter.view_isometric()
    return plotter

st.title("3D Model Gallery")

st.markdown(
    """
    All house massing models used for data collection can be found here!
    Select a model from the dropdown menu below to view it.
    """
)

try:
    curvilinear_files = sorted(
        [f for f in os.listdir(CURVILINEAR_DIR) if f.endswith((".stl", ".obj"))],
        key=extract_number
    )
except FileNotFoundError:
    st.error(f"Directory not found: {CURVILINEAR_DIR}")
    curvilinear_files = []

try:
    rectilinear_files = sorted(
        [f for f in os.listdir(RECTILINEAR_DIR) if f.endswith((".stl", ".obj"))],
        key=extract_number
    )
except FileNotFoundError:
    st.error(f"Directory not found: {RECTILINEAR_DIR}")
    rectilinear_files = []

model_options = [
    {"label": file, "path": os.path.join(CURVILINEAR_DIR, file)}
    for file in curvilinear_files
] + [
    {"label": file, "path": os.path.join(RECTILINEAR_DIR, file)}
    for file in rectilinear_files
]

selected_model = st.selectbox(
    "Select a model to view:",
    model_options,
    format_func=lambda option: option["label"] if option else "No models available"
)

if selected_model:
    file_path = selected_model["path"]
    with st.spinner(f"Loading model: {selected_model['label']}"):
        plotter = render_model(file_path)
        if plotter is not None:
            stpyvista(plotter)
        else:
            st.error(f"Failed to load model: {selected_model['label']}")
else:
    st.info("No models available to display.")
