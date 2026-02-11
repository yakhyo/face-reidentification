# Real-Time Face Re-Identification with FAISS, ArcFace & SCRFD

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-reidentification)](https://github.com/yakhyo/face-reidentification/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Docs-blue)](https://deepwiki.com/yakhyo/face-reidentification)



<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/user-attachments/assets/16d63ac6-57a4-464b-8d82-948e1a06b6e3" muted="false" width="100%"></video>

## Key Features

- **Real-Time Face Recognition**: Live processing from webcam or video files with optimized performance
- **Production-Ready Accuracy**: State-of-the-art SCRFD + ArcFace models for reliable detection and recognition
- **Intelligent Processing**: Smart batch optimization that adapts to face count for optimal performance
- **Scalable Vector Search**: FAISS-powered similarity search with native batch operations
- **Flexible Model Selection**: Multiple model sizes available to balance speed vs accuracy for your use case
- **Lightweight & Efficient**: Minimal dependencies with no threading overhead for maximum throughput

## Performance Optimizations

- **Native Batch Processing**: All face queries are stacked and passed to FAISS in a single vectorised call
- **Zero Threading Overhead**: CPU-bound FAISS operations run directly without GIL contention or lock overhead
- **Memory Efficient**: Minimal allocations with numpy-vectorised normalisation
- **Optimized for Video**: Designed for typical video scenarios (1-5 faces per frame)

> [!NOTE]
> Place your target face images in the `assets/faces/` directory. The filenames will be used as identity labels during recognition.

## Architecture

The system combines three powerful components:
1. **SCRFD** ([Paper](https://arxiv.org/abs/2105.04714)): Efficient face detection
2. **ArcFace** ([Paper](https://arxiv.org/abs/1801.07698)): Robust face recognition
3. **FAISS**: Fast similarity search for face re-identification

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
│   |── in_video.mp4
|   └── faces/           # Place target face images here
│     ├── face1.jpg
│     ├── face2.jpg
│     └── ...
├── database/           # FAISS database implementation
├── models/            # Neural network models
├── weights/           # Model weights (download required)
├── utils/            # Helper functions
├── main.py           # Main application entry
└── requirements.txt  # Dependencies
```

## Getting Started

### Prerequisites

> [!IMPORTANT]
> Make sure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yakyo/face-reidentification.git
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
| `--det-weight` | Detection model path | `weights/det_10g.onnx` |
| `--rec-weight` | Recognition model path | `weights/w600k_mbf.onnx` |
| `--similarity-thresh` | Face similarity threshold | `0.4` |
| `--confidence-thresh` | Detection confidence threshold | `0.5` |
| `--faces-dir` | Target faces directory | `assets/faces` |
| `--source` | Video source (file or camera index) | `0` |
| `--max-num` | Max faces per frame | `5` |
| `--db-path` | Custom database storage location | `./database/face_database` |
| `--update-db` | Force rebuild face database | `False` |
| `--output` | Specify output video path | `output_video.mp4` |

## Technical Notes

### Database Behavior
- Face database automatically saves/loads from disk
- Batch queries are handled in a single FAISS call (no per-face loop)
- Vectorised L2-normalisation via numpy for fast embedding preparation

### Performance Tips
- **Batch FAISS search**: All detected faces are queried in a single `index.search()` call
- **GPU acceleration**: Install `onnxruntime-gpu` for faster model inference
- **Database persistence**: Face database automatically saves/loads, avoiding rebuild on restart

## References

> [!NOTE]
> This project builds upon the following research:

1. [SCRFD: Sample and Computation Redistribution for Efficient Face Detection](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
2. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

<!-- ## Support

If you find this project useful, please consider giving it a star on GitHub! -->

