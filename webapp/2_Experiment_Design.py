import streamlit as st

st.title("📜 Experimental Design")

st.markdown(
    """
    The objective of the experiment is to collect a dataset of eye tracking data and EEG signals of people observing objects. This data will be used to train a machine learning model that predicts visual saliency. The model is to be used as a tool to facilitate better communication between designers and non-designers.

    Participants are first briefed on the experiment’s purpose and procedure. They are introduced to the equipment, including an EEG device and a HoloLens.

    The EEG device and HoloLens are fitted to the participants, and the setup is verified to ensure the equipment is functioning correctly. Participants then receive a tutorial on operating the HoloLens to familiarize them with its controls and interface.
    """
)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("assets/pictures/equipment_setup.png", caption="Example of Subject with EEG and Hololens Set Up")

st.markdown(
    """
    After completing the setup and tutorial, participants begin the main experiment. They are guided with the following prompt:

    > Imagine you are looking through a moodboard to design your own house.

    This prompt is designed to give participants a sense of purpose and encourage them to engage emotionally with the models without biasing their responses toward specific outcomes.

    During the experiment, participants view **30 models** presented in a randomized order (see the models [here](3D_Model_Gallery)). Each model is displayed for 1 minute and 15 seconds, during which participants are free to interact with the model, which includes scaling, rotating, and moving the object. To help participants manage their time, an alarm sounds 5 seconds before the end of each viewing session. The current model then automatically transitions to the next. Participants are also allowed to pause the experiment at any point of time.
    """)