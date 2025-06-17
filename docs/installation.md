# Installation Guide

This guide will walk you through the process of setting up the Face Re-Identification project on your system.

## Prerequisites

- Python 3.6 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Git

## Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yakyo/face-reidentification.git
   cd face-reidentification
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - ONNX and ONNX Runtime for model inference
   - OpenCV for image processing
   - NumPy for numerical operations
   - scikit-image for image transformations
   - FAISS for efficient similarity search

3. **Download Model Weights**

   You have two options for downloading the model weights:

   **Option 1**: Automated Download (Linux/macOS)
   ```bash
   sh download.sh
   ```

   **Option 2**: Manual Download
   Download the following weights and place them in the `weights` directory:

   | Model | File | Size | Purpose |
   |-------|------|------|---------|
   | SCRFD 500M | det_500m.onnx | 2.41 MB | Lightweight face detection |
   | SCRFD 2.5G | det_2.5g.onnx | 3.14 MB | Medium face detection |
   | SCRFD 10G | det_10g.onnx | 16.1 MB | High-accuracy face detection |
   | ArcFace MobileFace | w600k_mbf.onnx | 12.99 MB | Efficient face recognition |
   | ArcFace ResNet-50 | w600k_r50.onnx | 166 MB | High-accuracy face recognition |

## Project Structure

After installation, ensure you have the following directory structure:

```
face-reidentification/
├── assets/
│   ├── demo.mp4          # Sample video for testing
│   └── faces/            # Sample face images
├── database/             # Face database implementation
├── models/               # Model implementations
├── utils/               # Helper functions
└── weights/             # Downloaded model weights
```

## Verify Installation

1. Add sample faces to the database:
   ```bash
   # Copy face images to assets/faces/ directory
   # Then update the face database
   python main.py --faces-dir assets/faces --update-db
   ```

2. Run a test video:
   ```bash
   python main.py --source assets/demo.mp4 --output result.mp4
   ```

## Troubleshooting

Common issues and solutions:

1. **CUDA/GPU Issues**
   - Ensure CUDA is properly installed
   - Check if onnxruntime-gpu is installed correctly
   - Verify GPU drivers are up to date

2. **Model Loading Issues**
   - Verify all model files are in the `weights` directory
   - Check file permissions
   - Ensure model files are not corrupted

3. **Dependency Conflicts**
   - Create a new virtual environment
   - Install dependencies in order
   - Check for version conflicts in requirements.txt

For additional help, please refer to the project's GitHub issues page.
