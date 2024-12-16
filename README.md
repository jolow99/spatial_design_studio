# Waves: Predicting Spatial Perception with AI

A research project exploring how artificial intelligence can predict and analyze spatial perception in architectural design, developed in collaboration with HKS Singapore.

## Overview

This project investigates whether there are measurable differences in how design-trained and non-design-trained individuals perceive architectural spaces. Using eye tracking and EEG data, we developed AI models to predict spatial saliency and preference patterns.

## Repository Structure

The repository contains four main folders:

### 1. Unity Project Setup
- Contains Unity-based experiment setup for data collection
- Implements eye tracking and EEG integration with HoloLens 2
- Includes 3D architectural models for testing

### 2. Saliency Computation
- Processes raw eye tracking and EEG data
- Computes combined saliency scores
- Implements global processing for point cloud mapping

### 3. AI Pipeline
- Contains DGCNN-based model architecture
- Implements custom loss functions for saliency prediction
- Includes training and evaluation scripts
- Features both personal and demographic models

### 4. Webapp
- Streamlit-based web application
- Allows users to upload and analyze 3D models
- Visualizes saliency predictions across different demographics

## Authors

- Joseph Low 
- Rio Chan
- Tan Kay Wee 
- Charlene Teo En 
- Jay Edward Goh 

## Acknowledgments

Special thanks to HKS Singapore for their collaboration and support throughout this research project.
