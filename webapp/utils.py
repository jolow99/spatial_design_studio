import os
import streamlit as st
import pandas as pd
import re

@st.cache_data
def load_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

def get_subdirectories(directory):
    try:
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    except FileNotFoundError:
        st.error(f"Directory not found: {directory}")
        return []
    
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def get_files_in_directory(directory):
    try:
        return [f for f in os.listdir(directory) if f.endswith(".csv")]
    except FileNotFoundError:
        st.error(f"Directory not found: {directory}")
        return []