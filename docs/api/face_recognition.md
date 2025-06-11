# Face Recognition API Reference

This document details the ArcFace implementation used for face recognition in the project.

## ArcFace Class

### Overview

```python
from models import ArcFace
```

The ArcFace class implements face recognition using the ArcFace model, based on the paper "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".

### Constructor

```python
def __init__(self, model_path: str = None, session=None) -> None
```

#### Parameters
- `model_path` (str, optional): Path to the ONNX model file
- `session` (onnxruntime.InferenceSession, optional): Existing ONNX session

### Methods

#### get_feat
```python
def get_feat(self, images: np.ndarray) -> np.ndarray
```

Extracts face features from aligned face images.

##### Parameters
- `images` (numpy.ndarray): Input face image(s)

##### Returns
- numpy.ndarray: Face embedding features

#### __call__
```python
def __call__(self, image, kps)
```

Main inference method for feature extraction.

##### Parameters
- `image` (numpy.ndarray): Input image
- `kps` (numpy.ndarray): Face keypoints for alignment

##### Returns
- numpy.ndarray: Face embedding vector

### Example Usage

```python
# Initialize recognizer
recognizer = ArcFace(model_path="weights/w600k_r50.onnx")

# Process detected face
image = cv2.imread("face.jpg")
kps = ...  # Keypoints from SCRFD detector
embedding = recognizer(image, kps)

# Compare faces
similarity = compute_similarity(embedding1, embedding2)
```

### Implementation Details

1. **Input Processing**
   - Normalization: mean=127.5, std=127.5
   - Image alignment using keypoints
   - RGB color order (swapRB=True)

2. **Model Variants**
   - w600k_r50.onnx: ResNet-50 backbone (166 MB)
   - w600k_mbf.onnx: MobileFace backbone (12.99 MB)

3. **Feature Vector**
   - High-dimensional face embedding
   - Normalized for cosine similarity comparison

### Best Practices

1. **Model Selection**
   - Use ResNet-50 for highest accuracy
   - Use MobileFace for resource-constrained environments

2. **Input Requirements**
   - Properly aligned face images
   - Consistent image size and format
   - Good quality face images for enrollment

3. **Similarity Computation**
   - Use cosine similarity for comparison
   - Typical threshold range: 0.3-0.5
   - Adjust based on use case requirements
