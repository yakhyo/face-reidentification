# Face Re-Identification with SCRFD and ArcFace

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total) [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)

<video controls autoplay loop src="https://github.com/yakhyo/face-reidentification/assets/28424328/441880b0-1e43-4c28-9f63-b32bc9b6e6b4" muted="false" width="100%"></video>

This repository implements face re-identification using SCRFD for face detection and ArcFace for face recognition. It supports inference from webcam or video sources.

## Features (Updated on: 2024.07.29)

- [x] Smaller versions of SCFRD face detection model has been added
- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection. (Updated on: 2024.07.29)
  - Added models: SCRFD 500M (2.41 MB), SCRFD 2.5G (3.14 MB)
- [x] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition. (Updated on: 2024.07.29)
  - Added models: ArcFace MobileFace (12.99 MB)
- [x] **Real-Time Inference**: Supports both webcam and video file input for real-time processing.

Project folder structure:

```
├── assets/
│   ├── demo.mp4
│   └── in_video.mp4
├── faces/
│   ├── face1.jpg
│   ├── face2.jpg
│   └── ...
├── models/
│   ├── __init__.py
│   ├── scrfd.py
│   └── arcface.py
├── weights/
│   ├── det_10g.onnx
│   ├── det_2.5g.onnx
│   ├── det_500m.onnx
│   ├── w600k_r50.onnx
│   └── w600k_mbf.onnx
├── utils/
│   └── helpers.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakyo/face-reidentification.git
cd face-reidentification
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download weight files:

   a) Download weights from following links:

   | Model              | Weights                                                                                                   | Size     |
   | ------------------ | --------------------------------------------------------------------------------------------------------- | -------- |
   | SCRFD 500M         | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx)   | 2.41 MB  |
   | SCRFD 2.5G         | [det_2.5g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx)   | 3.14 MB  |
   | SCRFD 10G          | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx)     | 16.1 MB  |
   | ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 12.99 MB |
   | ArcFace ResNet-50  | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB   |

   |

   b) Run below command to download weights to `weights` directory (linux):

   ```bash
   sh download.sh
   ```

4. Put target faces into `faces` folder

```
faces/
    ├── name1.jpg
    ├── name2.jpg
```

Those file names will be displayed while real-time inference.

## Usage

```bash
python main.py --source assets/in_video.mp4
```

`main.py` arguments:

```
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT] [--similarity-thresh SIMILARITY_THRESH] [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE] [--max-num MAX_NUM] [--log-level LOG_LEVEL]

Face Detection-and-Recognition

options:
  -h, --help            show this help message and exit
  --det-weight DET_WEIGHT
                        Path to detection model
  --rec-weight REC_WEIGHT
                        Path to recognition model
  --similarity-thresh SIMILARITY_THRESH
                        Similarity threshold between faces
  --confidence-thresh CONFIDENCE_THRESH
                        Confidence threshold for face detection
  --faces-dir FACES_DIR
                        Path to faces stored dir
  --source SOURCE       Video file or video camera source. i.e 0 - webcam
  --max-num MAX_NUM     Maximum number of face detections from a frame
  --log-level LOG_LEVEL
                        Logging level
```

## Reference

1. https://github.com/deepinsight/insightface/tree/master/detection/scrfd
2. https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
