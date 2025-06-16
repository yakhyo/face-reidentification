# Face Detection API Reference

This document provides detailed information about the SCRFD (Sample and Computation Redistribution for Efficient Face Detection) implementation in the project.

## SCRFD Class

### Overview

```python
from models import SCRFD
```

The SCRFD class implements the face detection model based on the paper "Sample and Computation Redistribution for Efficient Face Detection".

### Constructor

```python
def __init__(self, 
             model_path: str,
             input_size: Tuple[int] = (640, 640),
             conf_thres: float = 0.5,
             iou_thres: float = 0.4) -> None
```

#### Parameters
- `model_path` (str): Path to the ONNX model file
- `input_size` (tuple): Input image size (default: (640, 640))
- `conf_thres` (float): Confidence threshold for detections (default: 0.5)
- `iou_thres` (float): IOU threshold for NMS (default: 0.4)

### Methods

#### detect
```python
def detect(self, image, max_num=0, metric="max")
```

Detects faces in an image and returns coordinates in the original image space.

##### Parameters
- `image` (numpy.ndarray): Input image in BGR format
- `max_num` (int): Maximum number of detections (0 for no limit)
- `metric` (str, optional): Metric for selecting faces when max_num is specified. Options: 
  - "max": Selects faces with the largest area
  - "center": Selects faces based on a combined score of area and distance from the image center. Faces closer to the center with reasonable size are prioritized over larger faces at the edges.
  Default: "max".

##### Returns
- tuple: (detections, keypoints)
  - detections: numpy.ndarray of shape (N, 5) with [x1, y1, x2, y2, score] in original image coordinates
  - keypoints: numpy.ndarray of shape (N, 5, 2) with facial landmarks in original image coordinates

#### forward
```python
def forward(self, image, threshold)
```

Internal method for model inference.

##### Parameters
- `image` (numpy.ndarray): Preprocessed input image
- `threshold` (float): Detection confidence threshold

##### Returns
- tuple: (scores_list, bboxes_list, kpss_list)

#### nms
```python
def nms(self, dets, iou_thres)
```

Non-maximum suppression for detections.

##### Parameters
- `dets` (numpy.ndarray): Detections array
- `iou_thres` (float): IOU threshold

##### Returns
- list: Indices of kept detections

### Example Usage

```python
# Initialize detector
detector = SCRFD(
    model_path="weights/det_10g.onnx",
    input_size=(640, 640),
    conf_thres=0.5
)

# Process image
image = cv2.imread("image.jpg")
boxes, points = detector.detect(image)

# Process detections
for box, points in zip(boxes, points):
    x1, y1, x2, y2, score = box.astype(np.int32)
    # Draw or process detection results
```

### Implementation Details

The detector uses a multi-scale feature pyramid network with the following characteristics:

- Feature stride FPN: [8, 16, 32]
- Number of anchors: 2
- Uses keypoints by default
- Input normalization: mean=127.5, std=128.0

### Performance Considerations

1. **Model Variants**
   - det_500m.onnx: Lightweight model (2.41 MB)
   - det_2.5g.onnx: Medium model (3.14 MB)
   - det_10g.onnx: High-accuracy model (16.1 MB)

2. **Input Size**
   - Default 640x640 provides good balance
   - Can be adjusted for speed/accuracy tradeoff

3. **Thresholds**
   - conf_thres: Controls false positive rate
   - iou_thres: Affects detection merging
