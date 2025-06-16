# Utilities API Reference

This document describes the utility functions provided in the `utils/` module.

## Logging Module (`utils/logging.py`)

### setup_logging
```python
def setup_logging(level=logging.INFO, log_to_file=False, filename="app.log") -> None
```

Sets up the logging configuration for the application.

**Parameters:**
- `level` (int): Logging level (default: logging.INFO)
- `log_to_file` (bool): Whether to log to a file (default: False)
- `filename` (str): Log file name when log_to_file is True (default: "app.log")

**Format:**
```
{timestamp} - {module_name} - {level} - {message}
```

## Face Alignment Functions (`utils/helpers.py`)

### estimate_norm
```python
def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]
```

Estimates the normalization transformation matrix for facial landmarks.

**Parameters:**
- `landmark` (np.ndarray): Array of shape (5, 2) representing facial landmarks
- `image_size` (int): Output image size, must be multiple of 112 or 128 (default: 112)

**Returns:**
- Tuple[np.ndarray, np.ndarray]: (transformation_matrix, inverse_transformation_matrix)

**Reference Landmarks:**
```python
reference_alignment = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)
```

### face_alignment
```python
def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112, mode: str = 'arcface') -> np.ndarray
```

Aligns a face image using detected landmarks.

**Parameters:**
- `image` (np.ndarray): Input image
- `landmark` (np.ndarray): Face landmarks (5,2)
- `image_size` (int): Output image size (default: 112)
- `mode` (str): Alignment mode, 'arcface' supported

**Returns:**
- np.ndarray: Aligned face image

### compute_similarity
```python
def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

Computes cosine similarity between two embeddings.

**Parameters:**
- `embedding1` (np.ndarray): First embedding vector
- `embedding2` (np.ndarray): Second embedding vector

**Returns:**
- float: Similarity score (0-1 range)

### Drawing Functions

#### draw_bbox
```python
def draw_bbox(image: np.ndarray, bbox: np.ndarray, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray
```

Draws a bounding box on an image.

**Parameters:**
- `image` (np.ndarray): Input image
- `bbox` (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
- `color` (tuple): RGB color (default: green)
- `thickness` (int): Line thickness (default: 2)

**Returns:**
- np.ndarray: Image with drawn bounding box

#### draw_bbox_info
```python
def draw_bbox_info(image: np.ndarray, bbox: np.ndarray, name: str, similarity: float) -> np.ndarray
```

Draws bounding box with identity information.

**Parameters:**
- `image` (np.ndarray): Input image
- `bbox` (np.ndarray): Bounding box coordinates
- `name` (str): Person's name
- `similarity` (float): Similarity score

**Returns:**
- np.ndarray: Annotated image

## Usage Examples

### Logging Configuration
```python
from utils.logging import setup_logging

# Console-only logging
setup_logging(level=logging.DEBUG)

# File and console logging
setup_logging(level=logging.INFO, log_to_file=True, filename="face_reid.log")
```

### Face Alignment
```python
from utils.helpers import face_alignment, estimate_norm

# Get transformation matrix
matrix, inverse_matrix = estimate_norm(landmarks, image_size=112)

# Align face image
aligned_face = face_alignment(image, landmarks, image_size=112)
```

### Visualization
```python
from utils.helpers import draw_bbox, draw_bbox_info

# Draw simple bounding box
image = draw_bbox(image, bbox, color=(0, 255, 0))

# Draw box with identity info
image = draw_bbox_info(image, bbox, name="John", similarity=0.95)
```
