# AI Pipeline for Attention Analysis

This pipeline processes and analyzes attention data from eye-tracking (ET) and EEG measurements, using a modified DGCNN (Dynamic Graph CNN) architecture.

## Directory Structure

ai-pipeline/
├── configs/
│   └── config.yaml          # Configuration settings
├── misc/
│   ├── dataset_analysis/    # Analysis scripts
│   └── dataset_processing/  # Data preprocessing scripts
├── src/                     # Source code
├── data/                    # Data directory (not included)
├── checkpoints/             # Model checkpoints
├── main.py                  # Main training script
├── plot_losses.py          # Utility for visualizing training losses
└── requirements.txt        # Dependencies

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data directory with the following structure:
```
data/
├── expert/
│   ├── subject_1/
│   │   ├── et/
│   │   └── eeg/
│   ├── subject_2/
│   └── subject_3/
└── novice/
    ├── subject_1/
    ├── subject_2/
    └── subject_3/
```

## Configuration

Edit `configs/config.yaml` to set your parameters:
```yaml
data:
  path: "data"
  subject_type: "expert"  # Options: "novice" or "expert"
  subject_id: "3"         # Options: "1", "2", "3"
  config_type: "eeg"      # Options: "et", "eeg"

model:
  k: 12                   # k-nearest neighbors
  dropout: 0.0
  gamma: 5.0

training:
  epochs: 30
  batch_size: 1
  learning_rate: 0.0003
  checkpoint_dir: "checkpoints/"
```

## Running the Pipeline

1. Train the model:
```bash
python main.py
```

2. Visualize training losses:
```bash
python plot_losses.py
```

## Analysis Tools

The `misc/dataset_analysis/` directory contains several analysis scripts:

- `analyze_attention_intensity.py`: Analyzes attention intensity distributions
- `analyze_dataset_by_model.py`: Compares data across different models
- `analyze_dataset_distribution.py`: Analyzes score distributions
- `analyze_form_type.py`: Compares curvilinear vs. rectilinear forms
- `analyze_spatial_attention.py`: Analyzes spatial patterns of attention
- `visualize_dataset.py`: Creates 3D visualizations of the point clouds

To run any analysis:
```bash
python -m misc.dataset_analysis.analyze_attention_intensity
```

## Data Processing

The `misc/dataset_processing/` directory contains utilities for data preprocessing:

- `normalize_eegScore.py`: Normalizes EEG scores
- `filenames.py`: Standardizes file naming conventions

## Output

- Training checkpoints are saved in `checkpoints/`
- Analysis results are saved in their respective analysis directories
- Visualizations are generated in the corresponding analysis folders

## Notes

- Ensure your data follows the expected directory structure
- The pipeline supports both ET and EEG data analysis
- Model checkpoints are automatically saved during training
- Analysis scripts generate both visualizations and statistical results
