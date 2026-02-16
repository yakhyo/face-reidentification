# Real-Time Face Re-Identification with FAISS, ArcFace & SCRFD

[![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total?color=blue&label=Downloads)](https://github.com/yakhyo/face-reidentification/releases)
[![GitHub Repo Stars](https://img.shields.io/github/stars/yakhyo/face-reidentification)](https://github.com/yakhyo/face-reidentification/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Docs-blue)](https://deepwiki.com/yakhyo/face-reidentification)

> [!TIP]
> The models and functionality in this repository are **integrated into [UniFace](https://github.com/yakhyo/uniface)** — an all-in-one face analysis library.
> [![PyPI Version](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/) [![GitHub Stars](https://img.shields.io/github/stars/yakhyo/uniface)](https://github.com/yakhyo/uniface/stargazers) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/user-attachments/assets/16d63ac6-57a4-464b-8d82-948e1a06b6e3" muted="false" width="100%"></video>

## Key Features

- **Real-Time Face Recognition**: Process webcam or video files with SCRFD detection and ArcFace embeddings
- **FAISS Similarity Search**: Batch cosine-similarity lookup using a FAISS inner-product index
- **Multiple Model Sizes**: Choose from lightweight to high-accuracy detection and recognition models
- **Minimal Dependencies**: Built on ONNX Runtime, OpenCV, NumPy, and FAISS with no extra frameworks

> [!NOTE]
> Place your target face images in the `assets/faces/` directory. The filenames will be used as identity labels during recognition.

## Components

1. **SCRFD** — Sample and Computation Redistribution for Efficient Face Detection
2. **ArcFace** — Additive Angular Margin Loss for Deep Face Recognition
3. **FAISS** — Facebook AI Similarity Search

### Available Models

| Category | Model | Size | Description |
|----------|-------|------|-------------|
| Detection | SCRFD 500M | 2.41 MB | Lightweight face detection |
| Detection | SCRFD 2.5G | 3.14 MB | Balanced performance |
| Detection | SCRFD 10G | 16.1 MB | High accuracy |
| Recognition | ArcFace MobileFace | 12.99 MB | Mobile-friendly recognition |
| Recognition | ArcFace ResNet-50 | 166 MB | High-accuracy recognition |

## Project Structure

```
├── assets/
│   ├── demo.mp4
│   ├── in_video.mp4
│   └── faces/              # Place target face images here
│       ├── face1.jpg
│       ├── face2.jpg
│       └── ...
├── database/               # FAISS database implementation
├── models/                 # Neural network models
├── weights/                # Model weights (download required)
├── utils/                  # Helper functions
├── main.py                 # Main application entry
└── requirements.txt        # Dependencies
```

## Getting Started

### Prerequisites

> [!IMPORTANT]
> Make sure you have Python 3.10+ installed on your system.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yakhyo/face-reidentification.git
cd face-reidentification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download model weights:**

<details>
<summary>Click to see download links 📥</summary>

| Model | Download Link | Size |
|-------|--------------|------|
| SCRFD 500M | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx) | 2.41 MB |
| SCRFD 2.5G | [det_2.5g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx) | 3.14 MB |
| SCRFD 10G | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx) | 16.1 MB |
| ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 12.99 MB |
| ArcFace ResNet-50 | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB |

</details>

**Quick download (Linux/Mac):**
```bash
sh download.sh
```

4. **Add target faces:**
Place face images in `assets/faces/` directory. The filename will be used as the person's identity.

## Usage

### Basic Usage
```bash
python main.py --source assets/in_video.mp4
```

### Command Line Arguments

> [!TIP]
> Use these arguments to customize the recognition behavior:

```bash
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT] 
               [--similarity-thresh SIMILARITY_THRESH] [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE] [--max-num MAX_NUM]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--det-weight` | Detection model path | `./weights/det_10g.onnx` |
| `--rec-weight` | Recognition model path | `./weights/w600k_mbf.onnx` |
| `--similarity-thresh` | Face similarity threshold | `0.4` |
| `--confidence-thresh` | Detection confidence threshold | `0.5` |
| `--faces-dir` | Target faces directory | `./assets/faces` |
| `--source` | Video source (file or camera index) | `./assets/in_video.mp4` |
| `--max-num` | Max faces per frame (0 = unlimited) | `0` |
| `--db-path` | Custom database storage location | `./database/face_database` |
| `--update-db` | Force rebuild face database | `False` |
| `--output` | Specify output video path | `output_video.mp4` |

## Technical Notes

- Face database is saved to and loaded from disk automatically; no rebuild needed on restart
- All detected faces in a frame are queried in a single FAISS `index.search()` call
- For GPU-accelerated inference, install `onnxruntime-gpu` instead of `onnxruntime`

## References

> [!NOTE]
> This project builds upon the following research:

1. [SCRFD: Sample and Computation Redistribution for Efficient Face Detection](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
2. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

<!-- ## Support

If you find this project useful, please consider giving it a star on GitHub! -->

