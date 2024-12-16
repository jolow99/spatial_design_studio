import streamlit as st
import pyvista as pv
import pandas as pd
import numpy as np
import tempfile
from io import StringIO
from stpyvista import stpyvista

# Utility: Load the mesh file
def load_mesh(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl" if uploaded_file.name.endswith(".stl") else ".obj") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        return pv.read(tmp_file_path)
    except Exception as e:
        st.error(f"Could not load the file: {e}")
        return None

# Utility: Generate a point cloud from the mesh
def generate_point_cloud(mesh, points_per_mesh=15000):
    """
    Generate a uniform point cloud from a mesh using triangle sampling without subdivision.
    Optionally center the point cloud at the origin.
    """
     # Ensure the mesh is triangulated
    mesh = mesh.triangulate()

    # Extract the vertices and faces
    vertices = mesh.points  # Get the mesh vertices
    faces = mesh.faces.reshape((-1, 4))[:, 1:]  # Extract triangular faces

    # Calculate total surface area and collect triangle data
    total_area = 0
    triangle_data = []
    for face in faces:
        v1, v2, v3 = vertices[face]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))  # Calculate the triangle area
        if area > 0.0001:  # Filter out degenerate triangles
            triangle_data.append((v1, v2, v3, area))
            total_area += area

    # Normalize the area to create a probability distribution
    areas = np.array([data[3] for data in triangle_data])
    probabilities = areas / areas.sum()

    # Generate points
    sampled_points = []
    for _ in range(points_per_mesh):
        # Randomly choose a triangle based on area weighting
        triangle_index = np.random.choice(len(triangle_data), p=probabilities)
        v1, v2, v3, _ = triangle_data[triangle_index]

        # Sample a point uniformly within the triangle
        u = np.sqrt(np.random.rand())  # Square root for uniform distribution across the area
        v = np.random.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v

        w = 1 - u - v
        point = u * v1 + v * v2 + w * v3
        sampled_points.append(point)

    # Convert to DataFrame
    point_cloud = pd.DataFrame(sampled_points, columns=["x", "y", "z"])

    centroid = point_cloud.mean(axis=0)  # Compute the centroid of the point cloud
    point_cloud[["x", "y", "z"]] -= centroid  # Translate the point cloud to center it

    max_coordinate = point_cloud.abs().max().max()  # Find the maximum absolute coordinate value
    point_cloud[["x", "y", "z"]] /= max_coordinate  # Normalize the coordinates to [-1, 1]

    return point_cloud


st.title("Step 1: Convert Mesh to Point Cloud")

st.markdown("""
The first step in using our AI models to predict saliency is to convert your 3D mesh objects to a point cloud composed of XYZ coordinates. 
""")

uploaded_file = st.file_uploader("Upload a 3D file (.stl or .obj)", type=["stl", "obj"])

if uploaded_file:
    mesh = load_mesh(uploaded_file)

    if mesh:
        st.success("Mesh loaded successfully!")

        points_per_mesh = st.number_input("Number of points to generate", min_value=1000, max_value=50000, value=15000)

        generate_button = st.button("Generate Point Cloud")

        if generate_button:
            with st.spinner("Generating point cloud..."):
                point_cloud = generate_point_cloud(mesh, points_per_mesh)
            st.success("Point cloud generated!")

            st.subheader("Generated Point Cloud")
            st.write(point_cloud.head())

            st.subheader("Point Cloud Visualization")
            plotter = pv.Plotter(border=False, window_size=[400, 400])
            polydata = pv.PolyData(point_cloud[["x", "y", "z"]].values)
            plotter.add_mesh(polydata, color="blue", point_size=5, render_points_as_spheres=True)
            plotter.view_isometric()
            stpyvista(plotter)

            st.subheader("Download Point Cloud")
            csv_buffer = StringIO()
            point_cloud.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Point Cloud as CSV",
                data=csv_buffer.getvalue(),
                file_name="point_cloud.csv",
                mime="text/csv",
            )

            st.markdown("""
                Now that you have your point cloud, you can proceed to the [next step](Run_Predictions).
            """)
