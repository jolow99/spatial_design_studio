# HoloLens Eye-Tracking Experiment Setup Guide

## Overview
This project implements an eye-tracking experiment system for the HoloLens 2, allowing researchers to collect precise gaze data while participants interact with 3D models. The system includes automatic model switching, data logging, and point cloud generation capabilities.

## Core Components

### Script Documentation

1. **ExperimentManager.cs**
   - Controls the experiment flow and model presentation
   - Manages countdown timers and model transitions
   - Handles user interaction through the "Ready" button
   - Coordinates with eye tracking system
   - Configurable settings for experiment timing and model display

2. **ExtendedEyeGazeDataProvider.cs**
   - Interfaces with HoloLens 2's eye-tracking hardware
   - Provides high-precision gaze data at 90Hz
   - Supports left eye, right eye, and combined gaze tracking
   - Handles eye-tracking permissions and initialization
   - Implements error handling and validation

3. **EyeTrackingExample.cs**
   - Records participant gaze data to CSV files
   - Maintains consistent 90Hz sampling rate
   - Transforms gaze points to local model space
   - Implements efficient data buffering and file I/O
   - Handles coordinate system transformations

4. **meshexporter_surface.cs**
   - Generates point cloud representations of 3D models
   - Exports coordinate data to CSV format
   - Uses surface-based sampling for uniform point distribution
   - Configurable point density settings
   - Supports batch processing of multiple models

5. **ModelPrefabCreator.cs**
   - Unity Editor tool for bulk prefab creation
   - Automates experiment setup process
   - Configurable default scale and rotation settings
   - Automatically populates ExperimentManager references
   - Streamlines project organization

## Setup Instructions

### Prerequisites
- Unity 2022.3 LTS or newer
- Visual Studio 2022 with UWP development workload
- Windows 10/11 SDK (10.0.19041.0 or newer)
- HoloLens 2 device with developer mode enabled

### Development Environment Setup

1. **Mixed Reality Toolkit Installation**
   ```
   a. Install the Mixed Reality Feature Tool:
      https://learn.microsoft.com/en-us/windows/mixed-reality/develop/unity/welcome-to-mr-feature-tool
   
   b. Use the tool to download:
      - Mixed Reality OpenXR Plugin
      - MRTK3 Foundation
      - MRTK3 Input
      - MRTK3 Standard Assets
   ```

2. **Unity Project Configuration**
   ```
   a. Create new Unity project (3D URP)
   b. Import MRTK packages
   c. Configure project settings following:
      https://learn.microsoft.com/en-us/windows/mixed-reality/mrtk-unity/mrtk3-overview/getting-started/setting-up/setup-new-project
   ```

### Project Setup

1. **Model Preparation**
   ```
   a. Import 3D models (.obj format) to Assets/Models/ExperimentObjects
   b. Create point cloud data:
      - Add empty GameObject to scene
      - Attach meshexporter_surface.cs
      - Configure point count in inspector
      - Use context menu to execute export
      - Point clouds will be saved to Assets/Models/PointClouds
   ```

2. **Experiment Setup**
   ```
   a. Create new empty GameObject named "ExperimentController"
   b. Attach scripts:
      - ExperimentManager.cs
      - ExtendedEyeGazeDataProvider.cs
      - EyeTrackingExample.cs
   ```

3. **UI Setup**
   ```
   a. Create MRTK button for "Ready"
   b. Add TextMeshPro elements:
      - Instructions text
      - Countdown text
   c. Configure audio:
      - Add AudioSource component
      - Import and assign countdown beep sound
   ```

4. **Prefab Creation**
   ```
   a. Open Tools > Bulk Model Prefab Creator
   b. Set source folder (Models/ExperimentObjects)
   c. Configure default scale/rotation
   d. Click "Create Prefabs"
   e. Click "Setup ExperimentManager"
   ```

5. **Scene Organization**
   ```
   a. Create spawn point GameObject
   b. Position at desired model presentation location
   c. Link references in ExperimentManager inspector:
      - Spawn Point
      - UI elements
      - Audio components
      - Prefabs (auto-populated)
   ```

### Building and Deployment

1. **Unity Build Settings**
   ```
   a. File > Build Settings
   b. Switch platform to Universal Windows Platform
   c. Configure settings:
      - Target Device: HoloLens
      - Architecture: ARM64
      - Build Type: D3D Project
      - Build and Run on: Local Machine
   ```

2. **Visual Studio Configuration**
   ```
   a. Open generated .sln file
   b. Set configuration to Release/ARM64
   c. Set device to Remote Machine or Device
   d. Enter HoloLens IP in debugging properties
   ```

3. **HoloLens Deployment**
   ```
   a. Enable Developer Mode on HoloLens
   b. Ensure device is on same network as PC
   c. Build and deploy from Visual Studio
   ```

## Data Collection

The system generates two types of data files:

1. **Eye Tracking Data**
   - Format: CSV
   - Fields: Timestamp, HitObject, HitPointX/Y/Z, NormalX/Y/Z
   - Location: HoloLens device storage
   - Naming: eyetrackingdata__{ModelName}.csv

2. **Point Cloud Data**
   - Format: CSV
   - Fields: x, y, z coordinates
   - Location: Project's PointClouds folder
   - Naming: {ModelName}_points.csv

## Best Practices

- Ensure consistent lighting conditions
- Calibrate eye tracking for each participant
- Maintain stable network connection during deployment
- Regular data backups from HoloLens device
- Monitor system performance during extended sessions

## Troubleshooting

Common issues and solutions:

1. **Eye Tracking Permission Denied**
   - Verify Research Mode is enabled
   - Check app capabilities in manifest
   - Restart HoloLens device

2. **Build Failures**
   - Verify SDK installation
   - Check platform tools installation
   - Clean solution and rebuild

3. **Data Recording Issues**
   - Check available device storage
   - Verify file permissions
   - Monitor debug output for errors

## Support

For additional assistance:
- MRTK Documentation: https://learn.microsoft.com/en-us/windows/mixed-reality/mrtk-unity/
- HoloLens Development: https://learn.microsoft.com/en-us/windows/mixed-reality/develop/
- Unity Forums: https://forum.unity.com/