# About This App

This webapp is built on Streamlit, designed to provide an interactive and user-friendly platform for exploring the outcomes and methodology of our work. It is divided into two sections: **Our Process** and **Try It Yourself**.

The **Our Process** section provides details about how we approached this project, including an overview of our experiment design, visualizations of the collected data, and the assets used during the experiments.

The **Try It Yourself** section allows you to upload a model and predict its saliency, enabling you to see how the saliency is predicted to vary across different areas of the design as well as across different people.

# How to Run the App

Follow these steps to set up and run the application locally:

## 1. Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- pip (Python's package installer)

You can verify these by running:

```
python --version
pip --version
```

## 2. Set Up the Virtual Environment

1.  Create a virtual environment:
```bash
python -m venv venv
```
2.  Activate the virtual environment using:
Windows

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

## 3. Install Dependencies    
Use requirements.txt to install the required Python packages:      

```
pip install -r requirements.txt
```

## 4. Run the Streamlit App
Start the app using the following command:

```
streamlit run Home.py
```
