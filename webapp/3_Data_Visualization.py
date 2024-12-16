import os
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import numpy as np
from utils import get_subdirectories, load_csv_file, natural_sort_key, get_files_in_directory
from constants import ET_COLOR_MAPPING, ET_CLASS_LABELS, EEG_COLOR_MAPPING, EEG_CLASS_LABELS, LIGHT_GREY

POINT_CLOUD_DIR = "assets/experiment_point_cloud_data"

@st.cache_data
def create_point_cloud(data, x_col, y_col, z_col):
    return pv.PolyData(data[[x_col, y_col, z_col]].values)

def convert_to_classes(normalized_scores, config_type):
    if config_type == 'eeg':
        color_mapping = EEG_COLOR_MAPPING
        class_labels = EEG_CLASS_LABELS
        
        classes = np.zeros_like(normalized_scores, dtype=int)
        classes[normalized_scores <= -0.5] = 0
        classes[(-0.5 < normalized_scores) & (normalized_scores <= 0)] = 1
        classes[normalized_scores == 0] = 2
        classes[(0 < normalized_scores) & (normalized_scores <= 0.5)] = 3
        classes[normalized_scores > 0.5] = 4
        
    elif config_type == 'et':
        color_mapping = ET_COLOR_MAPPING
        class_labels = ET_CLASS_LABELS
        
        classes = np.zeros_like(normalized_scores, dtype=int)
        classes[normalized_scores == 0] = 0
        classes[(0 < normalized_scores) & (normalized_scores <= 0.025)] = 1
        classes[(0.025 < normalized_scores) & (normalized_scores <= 0.050)] = 2
        classes[(0.050 < normalized_scores) & (normalized_scores <= 0.1)] = 3
        classes[normalized_scores > 0.1] = 4

    return classes, color_mapping, class_labels

def render_class_colored_point_cloud(cloud, classes, color_mapping):
    if not hasattr(cloud, "points") or cloud.points is None or len(cloud.points) == 0:
        raise ValueError("The provided cloud object does not contain valid points.")

    rgb_colors = np.array([color_mapping[int(cls)] for cls in classes], dtype=np.uint8)

    if len(rgb_colors) != len(cloud.points):
        raise ValueError("The number of RGB colors does not match the number of points in the cloud.")

    cloud["RGB"] = rgb_colors  

    plotter = pv.Plotter(border=False, window_size=[400, 400])
    plotter.add_mesh(
        cloud,
        scalars="RGB",  
        rgb=True,      
        point_size=5,
        render_points_as_spheres=True
    )

    plotter.camera_position = [
        (4, 1, 1),  
        (0, 0, 0),  
        (0, 1, 0)
    ]
    return plotter

# START: ACTUAL PAGE
st.title("Data Visualisation")
st.markdown(
"""
This section is for the exploration and analysis of the raw results collected from participants in our study. By selecting a subject and model, you can view various visualizations that illustrate where participants directed their attention.
"""
)

person_options = get_subdirectories(POINT_CLOUD_DIR)
if not person_options:
    st.error("No participants found in the data directory.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    selected_person = st.selectbox("Select a Person", sorted(person_options))

subfolder_path_et = os.path.join(POINT_CLOUD_DIR, selected_person, "ET")
subfolder_path_eeg = os.path.join(POINT_CLOUD_DIR, selected_person, "EEG")

file_options_et = get_files_in_directory(subfolder_path_et)
file_options_eeg = get_files_in_directory(subfolder_path_eeg)

common_files = sorted(set(file_options_et).intersection(set(file_options_eeg)), key=natural_sort_key)


if not common_files:
    st.warning(f"No common files found for {selected_person} in ET and EEG folders.")
    st.stop()

with col2:
    selected_file = st.selectbox("Select a File", list(common_files))

file_path_et = os.path.join(subfolder_path_et, selected_file)
file_path_eeg = os.path.join(subfolder_path_eeg, selected_file)

data_et = load_csv_file(file_path_et)
data_eeg = load_csv_file(file_path_eeg)

if data_et is not None and data_eeg is not None:
    st.subheader("Raw Hit Point Visualization")
    st.markdown(
    """
    This visualization represents the raw gaze data collected during the study, with each point on the model indicating a specific location where the participant directed their gaze.
    """
    )
    try:
        base_cloud_et = create_point_cloud(data_et, "x", "y", "z")
        salient_data_et = data_et[data_et["SaliencyScore"] != 0]
        salient_cloud_et = create_point_cloud(salient_data_et, "x", "y", "z")

        plotter_et = pv.Plotter(border=False, window_size=[400, 400])
        plotter_et.add_mesh(base_cloud_et, color=LIGHT_GREY, point_size=5, render_points_as_spheres=True)
        plotter_et.add_mesh(salient_cloud_et, color=[0, 0, 255], point_size=8, render_points_as_spheres=True)
        plotter_et.camera_position = [
            (4, 1, 1),
            (0, 0, 0),
            (0, 1, 0)
        ]

        stpyvista(plotter_et)

    except KeyError as e:
        st.warning(f"Could not render raw hit points: Missing column {e}")
    except ValueError as e:
        st.error(f"Error: {e}")

    st.divider()
        
    st.subheader("Attention Visualization")
    st.markdown(
    """
    This visualization illustrates the subject's focus on various parts of the model, determined **solely through eye-tracking data**. The level of attention is indicated by the accompanying legend.
    """
    )

    try:
        base_cloud_et = create_point_cloud(data_et, "x", "y", "z")
        classes_et, color_mapping_et, class_labels_et = convert_to_classes(data_et["NormalizedScore"].values, "et")
        plotter_et = render_class_colored_point_cloud(base_cloud_et, classes_et, color_mapping_et)

        base_cloud_eeg = create_point_cloud(data_eeg, "x", "y", "z")
        classes_eeg, color_mapping_eeg, class_labels_eeg = convert_to_classes(data_eeg["NormalizedEEGScore"].values, "eeg")
        plotter_eeg = render_class_colored_point_cloud(base_cloud_eeg, classes_eeg, color_mapping_eeg)

        stpyvista(plotter_et)
        st.markdown("**ET Legend**")
        for class_id, color in color_mapping_et.items():
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(
                f'<span style="color:{color_hex}; font-weight:bold;">&#9632;</span> {class_labels_et[class_id]}',
                unsafe_allow_html=True
            )

        st.divider()

        st.subheader("Preference Visualization")
        st.markdown(
        """
        This visualization highlights the subject's focus on different parts of the model, as well as their **emotional responses** while observing these areas. The analysis is based on a **combination of eye-tracking and EEG data**, with the accompanying legend indicating the various preferences.
        """
        )
        stpyvista(plotter_eeg)
        st.markdown("**EEG Legend**")
        for class_id, color in color_mapping_eeg.items():
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(
                f'<span style="color:{color_hex}; font-weight:bold;">&#9632;</span> {class_labels_eeg[class_id]}',
                unsafe_allow_html=True
            )

    except KeyError as e:
        st.warning(f"Could not render class-based points: Missing column {e}")
    except ValueError as e:
        st.error(f"Error: {e}")
