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

**Class Attributes:**
- `input_size`: (112, 112) - Required input image size
- `normalization_mean`: 127.5 - Normalization mean value
- `normalization_scale`: 127.5 - Normalization scale value

### Methods

#### get_embedding
```python
get_embedding(face_img: np.ndarray, facial_points: np.ndarray) -> np.ndarray
```

Extracts a face embedding from a face image using facial landmarks for alignment.

**Parameters:**
- `face_img` (np.ndarray): Face image in BGR format
- `facial_points` (np.ndarray): 5 facial landmarks of shape (5, 2) for face alignment

**Returns:**
- np.ndarray: 512-dimensional face embedding normalized to unit length

**Note:** The method handles:
- Image resizing to 112x112
- Normalization (subtract mean, divide by scale)
- Color format conversion if needed

**Note:** The `compute_similarity` function is not part of the ArcFace class, but is available as a utility function in `utils/helpers.py`:

```python
from utils.helpers import compute_similarity

# Usage
similarity = compute_similarity(embedding1, embedding2)
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

# Initialize model with CUDA support
model = ArcFace("weights/w600k_r50.onnx")  # Will use CUDA if available

# Prepare face image
face_img = cv2.imread("face.jpg")
face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

# Get face embedding (requires facial landmarks)
embedding = model.get_embedding(face_img, kps)

# Compare with another face
similarity = model.compute_similarity(embedding, other_embedding)
if similarity > 0.4:  # Default threshold
    print(f"Same person, similarity: {similarity:.2f}")
```

## Model Variants

### ResNet-50 Backbone
- **File**: w600k_r50.onnx
- **Size**: 166 MB
- **Features**:
  - High accuracy
  - 512-dimensional embeddings
  - Suitable for server deployments
  - Good for high-security applications

### MobileFace Backbone
- **File**: w600k_mbf.onnx
- **Size**: 12.99 MB
- **Features**:
  - Lightweight
  - 512-dimensional embeddings
  - Suitable for edge devices
  - Good for real-time applications

## Implementation Details

1. **Model Loading**
   - Uses ONNX Runtime for inference
   - Supports both CPU and CUDA execution providers
   - Automatic provider selection based on availability

2. **Image Preprocessing**
   - Automatic resizing to 112x112
   - Normalization with mean=127.5, scale=127.5
   - RGB color format handling

3. **Error Handling**
   - Input validation and shape checking
   - Comprehensive logging for debugging
   - Clear error messages for common issues

4. **Best Practices**
   - Use aligned face images (see utils.helpers.face_alignment)
   - Keep consistent image preprocessing
   - Validate similarity thresholds for your use case
   - Consider model size vs accuracy tradeoffs
