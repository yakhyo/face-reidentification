# Real-Time Face Re-Identification with FAISS, ArcFace & SCRFD

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-reidentification)](https://github.com/yakhyo/face-reidentification/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)

<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/yakhyo/face-reidentification/assets/28424328/441880b0-1e43-4c28-9f63-b32bc9b6e6b4" muted="false" width="100%"></video>

This repository implements face re-identification using SCRFD for face detection and ArcFace for face recognition. It supports inference from webcam or video sources with efficient face database management using FAISS.

## Features

- [x] **Face Detection**: Utilizes [SCRFD (Sample and Computation Redistribution for Efficient Face Detection)](https://arxiv.org/abs/2105.04714)
  - Multiple model variants for different use cases:
    - SCRFD 500M (2.41 MB): Lightweight, ideal for mobile/edge
    - SCRFD 2.5G (3.14 MB): Balanced performance
    - SCRFD 10G (16.1 MB): High accuracy
  
- [x] **Face Recognition**: Employs [ArcFace](https://arxiv.org/abs/1801.07698) with multiple backbones:
  - ResNet-50 (166 MB): High-accuracy recognition
  - MobileFace (12.99 MB): Efficient, mobile-friendly option

- [x] **Efficient Face Database**:
  - FAISS vector database integration for fast similarity search
  - Easy face enrollment and management
  - Persistent storage of face embeddings
  - Optimized for large-scale face recognition

- [x] **Enhanced Features**:
  - Comprehensive logging system with file and console output
  - Multiple operation modes (add, list, camera, video)
  - Configurable similarity and confidence thresholds
  - Real-time processing for both webcam and video files

## Project Structure

```
├── assets/                 # Project assets
│   ├── demo.mp4           # Demo video
│   ├── in_video.mp4       # Sample input video
│   └── faces/             # Face database images
│       ├── face1.jpg
│       └── ...
├── database/              # Face database implementation
│   ├── __init__.py
│   └── face_db.py        # FAISS-based face database
├── models/                # Model implementations
│   ├── __init__.py
│   ├── scrfd.py          # SCRFD face detection
│   └── arcface.py        # ArcFace recognition
├── utils/                 # Utility functions
│   ├── logging.py        # Logging configuration
│   └── helpers.py        # Helper functions
├── weights/               # Model weights
│   ├── det_10g.onnx      # SCRFD 10G
│   ├── det_2.5g.onnx     # SCRFD 2.5G
│   ├── det_500m.onnx     # SCRFD 500M
│   ├── w600k_r50.onnx    # ArcFace ResNet-50
│   └── w600k_mbf.onnx    # ArcFace MobileFace
└── docs/                  # Comprehensive documentation
    ├── installation.md    # Installation guide
    ├── usage.md          # Usage instructions
    ├── models.md         # Model details
    └── api/              # API documentation
```

## Quick Start

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Faces to Database**:
   ```bash
   python main.py --mode add --dir assets/faces/ --auto-name
   ```

3. **Run Recognition**:
   ```bash
   # Using webcam
   python main.py --mode camera --source 0

   # Using video file
   python main.py --mode video --input assets/demo.mp4 --output result.mp4
   ```

## Documentation

For detailed information, please refer to our [documentation](docs/README.md):
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [Model Details](docs/models.md)
- [API Reference](docs/api/README.md)
