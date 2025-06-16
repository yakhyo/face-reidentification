# Face Recognition (ArcFace) API Reference

## Overview

The Face Recognition module implements the ArcFace algorithm for face recognition and embedding generation. It provides high-accuracy face recognition capabilities with support for different model backbones.

## ArcFace Class

### Initialization
```python
ArcFace(model_path: str)
```

Creates a new ArcFace model instance.

**Parameters:**
- `model_path` (str): Path to ONNX model file (ResNet-50 or MobileFace)

### Methods

#### get_embedding
```python
get_embedding(face_img: np.ndarray) -> np.ndarray
```

Extracts a face embedding from an aligned face image.

**Parameters:**
- `face_img` (np.ndarray): Aligned face image (112x112 pixels)

**Returns:**
- np.ndarray: 512-dimensional face embedding

#### get_similarity
```python
get_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

Computes cosine similarity between two face embeddings.

**Parameters:**
- `embedding1` (np.ndarray): First face embedding
- `embedding2` (np.ndarray): Second face embedding

**Returns:**
- float: Similarity score between 0 and 1

## Usage Example

```python
from models.arcface import ArcFace
import cv2

# Initialize model
model = ArcFace("weights/w600k_r50.onnx")

# Load and preprocess face image
face_img = cv2.imread("face.jpg")
face_img = cv2.resize(face_img, (112, 112))

# Get face embedding
embedding = model.get_embedding(face_img)

# Compare with another face
similarity = model.get_similarity(embedding, other_embedding)
if similarity > 0.4:
    print("Same person")
```

## Model Variants

### ResNet-50 Backbone
- **File**: w600k_r50.onnx
- **Size**: 166 MB
- **Features**:
  - High accuracy
  - 512-dimensional embeddings
  - Suitable for server deployments

### MobileFace Backbone
- **File**: w600k_mbf.onnx
- **Size**: 12.99 MB
- **Features**:
  - Lightweight
  - 512-dimensional embeddings
  - Suitable for edge devices

## Best Practices

1. **Image Preprocessing**
   - Input images should be aligned using face landmarks
   - Use 112x112 pixel size
   - Convert to RGB format

2. **Similarity Thresholds**
   - Default: 0.4
   - Higher values (e.g., 0.6) for stricter matching
   - Lower values for more lenient matching

3. **Performance Optimization**
   - Batch processing when possible
   - GPU acceleration with ONNX Runtime
   - Proper error handling for invalid inputs
