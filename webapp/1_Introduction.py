import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

st.title("🌊 WAVES 🌊")

st.markdown("""
*WAVES* explores the integration of artificial intelligence (AI) into architectural design workflows to enhance communication between architects and clients by leveraging unconscious perceptions of space. Partnering with HKS Singapore, a global architecture and design firm, the research addresses the challenge of maintaining the intuitive and emotional aspects of design in an increasingly AI-driven environment. The project focuses on understanding whether AI can capture and predict unconscious spatial preferences using eye-tracking and EEG data.

To achieve this, an experimental methodology was developed to collect a dataset of unconscious responses to 3D architectural models. The experiment involved participants from both design-educated and non-design-educated backgrounds, using immersive tools such as the Microsoft HoloLens 2 and EEG devices to capture gaze patterns and brain activity. The collected data was processed to compute saliency scores, which quantify attention and emotional engagement with design elements, and analyzed to uncover differences in perceptions between the two groups.

Based on this research, a predictive AI model was developed to identify patterns of unconscious saliency across demographics. This AI tool facilitates communication by visually highlighting design elements likely to resonate with clients or elicit negative reactions, even when preferences are not explicitly expressed. Additionally, the project includes innovative applications of AI for moodboard generation, user persona development, and evaluation of design outcomes.

This work demonstrates the potential for AI to bridge the gap between human creativity and computational precision in architectural design, offering a data-driven foundation for more intuitive and effective client-architect collaboration. The findings also provide insights into how unconscious perceptions of design can enhance participatory workflows, paving the way for more personalized and user-centered design processes.        
""")

st.video("https://youtu.be/6XdvxqMONhg")

st.markdown("""
This app is divided into two sections: **Our Process** and **Try It Yourself**.

The **Our Process** section provides details about how we approached this project, including an overview of our [experiment design](Experiment_Design), [visualizations of the collected data](Data_Visualization), and the [assets used during the experiments](3D_Model_Gallery).

The **Try It Yourself** section allows you to upload a model and predict its saliency, enabling you to see how the saliency is predicted to vary across different areas of the design as well as across different people.
""")

st.divider()

main_container = st.empty()
main_container.empty()

with main_container.container():
    st.subheader("Learn the Controls")
    placeholder = st.empty()

    file_path = "assets/models/curvilinear/curved3.obj"

    plotter = pv.Plotter(border=False, window_size=[500, 400])
    plotter.background_color = "#f0f8ff"

    try:
        reader = pv.OBJReader(file_path)
        mesh = reader.read()
        plotter.add_mesh(mesh, color="orange", specular=0.5)
        plotter.view_isometric()

        with placeholder.container():
            stpyvista(plotter)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")


    with st.expander("🎮 Controls", expanded=True):
        controls_table = {
            "Control": [
                "LMB + Drag",
                "Ctrl + LMB + Drag",
                "Shift + Drag",
                "Scroll",
            ],
            "Description": [
                "Free rotate",
                "Rotate around center",
                "Pan",
                "Zoom",
            ],
        }
        st.dataframe(controls_table, use_container_width=True)
