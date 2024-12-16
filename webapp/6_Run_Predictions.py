import os
import torch
import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
from sklearn.neighbors import NearestNeighbors
from dgcnn import ModifiedDGCNN
from torch_geometric.data import Data
from utils import get_subdirectories
from constants import ET_COLOR_MAPPING, ET_CLASS_LABELS, EEG_COLOR_MAPPING, EEG_CLASS_LABELS

MODEL_BASE_PATH = "assets/ai_models"
LIGHT_GREY = [237, 237, 237]

@st.cache_resource
def load_model(model_path):
    """Load the saved model."""
    model = ModifiedDGCNN(num_classes=5, k=20)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def compute_geometric_features(points, k=20):
    """Compute geometric features for each point in the point cloud."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    N = points.shape[0]
    normals = np.zeros((N, 3))
    curvatures = np.zeros(N)
    densities = np.zeros(N)
    heights = np.zeros(N)
    
    for i in range(N):
        neighborhood = points[indices[i]]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        cov = np.dot(centered.T, centered) / (k - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
        densities[i] = np.mean(distances[i, 1:])
        heights[i] = points[i, 2] - np.mean(neighborhood[:, 2])
    
    features = np.column_stack([curvatures, densities, heights, normals])
    return features

def run_inference(model, points, geom_features):
    """Run inference on the uploaded point cloud with geometric features."""
    # Combine spatial and geometric features
    full_features = np.hstack([points, geom_features])
    tensor_input = torch.tensor(full_features, dtype=torch.float32).unsqueeze(0)
    
    data = Data(x=tensor_input[0])
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long) 
    
    with torch.no_grad():
        logits = model(data) 
    
    predicted_classes = np.argmax(logits.numpy(), axis=1)
    return predicted_classes

def create_class_colored_point_cloud(points, predicted_classes, color_mapping):
    """Create a point cloud with colors based on predicted classes."""
    colors = np.array([color_mapping[int(cls)] for cls in predicted_classes], dtype=np.uint8)

    cloud = pv.PolyData(points)
    cloud["RGB"] = colors 
    return cloud

def render_binned_point_cloud(cloud):
    """Render the point cloud with RGB class-based colors."""
    plotter = pv.Plotter(border=False, window_size=[400, 400])
    plotter.add_mesh(
        cloud,
        scalars=None, 
        rgb=True, 
        point_size=5,
        render_points_as_spheres=True
    )
    plotter.view_isometric()
    return plotter

def display_legend(color_mapping, class_labels):
    """Display a legend for class descriptions and their corresponding colors."""
    st.markdown("""**Class Legend**""")
    legend_items = ""
    for cls, color in color_mapping.items():
        rgb_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"  # Convert RGB to HEX
        legend_items += f"""
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
            <div style='width: 20px; height: 20px; background-color: {rgb_hex}; margin-right: 10px;'></div>
            <span style='font-size: 16px;'>Class {cls}: {class_labels[cls]}</span>
        </div>
        """
    st.markdown(legend_items, unsafe_allow_html=True)

def main():
    st.title("Step 2: Run Saliency Predictions")

    st.markdown(""" Our AI models are personalized, trained to predict saliency for each individual subject. You can choose to use the model of any experiment participant and upload your point cloud CSV. If you don’t have a point cloud CSV, you can proceed [here](Convert_Mesh_to_Point_Cloud) to generate one.

    The models will predict both Attention Saliency and Preference Saliency. Attention Saliency indicates the subject's focus on different areas of the model, while Preference Saliency reveals the subject's focus and their emotional responses to various parts of the model. """)
    people = get_subdirectories(MODEL_BASE_PATH)
    if not people:
        st.error("No models found in the assets/ai_models directory.")
        return

    selected_person = st.selectbox("Select a Person", sorted(people))

    person_model_path = os.path.join(MODEL_BASE_PATH, selected_person)
    et_model_path = os.path.join(person_model_path, "attention.pt")
    eeg_model_path = os.path.join(person_model_path, "preference.pt")

    # Ensure both ET and EEG models exist
    if not os.path.exists(et_model_path) or not os.path.exists(eeg_model_path):
        st.error("Both ET and EEG models must exist for the selected person.")
        return

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            if not all(col in data.columns for col in ["x", "y", "z"]):
                st.error("The uploaded file must contain 'x', 'y', and 'z' columns.")
                return
            
            points = data[["x", "y", "z"]].to_numpy()
            st.write("Uploaded point cloud:", data.head())
            
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            return

        with st.spinner("Computing geometric features..."):
            geom_features = compute_geometric_features(points)

        # Load models
        et_model = load_model(et_model_path)
        eeg_model = load_model(eeg_model_path)

        # Run inference for ET model
        with st.spinner("Running attention model inference..."):
            et_predicted_classes = run_inference(et_model, points, geom_features)

        # Run inference for EEG model
        with st.spinner("Running preference model inference..."):
            eeg_predicted_classes = run_inference(eeg_model, points, geom_features)

        st.divider()

        st.subheader("Attention Model Prediction")
        display_legend(ET_COLOR_MAPPING, ET_CLASS_LABELS)
        unique_classes_et, counts_et = np.unique(et_predicted_classes, return_counts=True)
        class_distribution_et = pd.DataFrame({
            "Class": unique_classes_et,
            "Point Count": counts_et
        })
        st.markdown("**Class Counts for Attention Model:**")
        st.table(class_distribution_et)

        cloud_et = create_class_colored_point_cloud(points, et_predicted_classes, ET_COLOR_MAPPING)
        stpyvista(render_binned_point_cloud(cloud_et))

        st.divider()

        st.subheader("Preference Model Prediction")
        display_legend(EEG_COLOR_MAPPING, EEG_CLASS_LABELS)
        unique_classes_eeg, counts_eeg = np.unique(eeg_predicted_classes, return_counts=True)
        class_distribution_eeg = pd.DataFrame({
            "Class": unique_classes_eeg,
            "Point Count": counts_eeg
        })
        st.markdown("**Class Counts for Preference Model:**")
        st.table(class_distribution_eeg)

        cloud_eeg = create_class_colored_point_cloud(points, eeg_predicted_classes, EEG_COLOR_MAPPING)
        stpyvista(render_binned_point_cloud(cloud_eeg))

        st.divider()

        st.subheader("Download the Predictions")

        et_results = data.copy()
        et_results["PredictedClass"] = et_predicted_classes
        et_csv = et_results.to_csv(index=False)
        st.download_button(
            label="Download ET Predictions as CSV",
            data=et_csv,
            file_name="et_predicted_classes.csv",
            mime="text/csv"
        )

        eeg_results = data.copy()
        eeg_results["PredictedClass"] = eeg_predicted_classes
        eeg_csv = eeg_results.to_csv(index=False)
        st.download_button(
            label="Download EEG Predictions as CSV",
            data=eeg_csv,
            file_name="eeg_predicted_classes.csv",
            mime="text/csv"
        )

main()
