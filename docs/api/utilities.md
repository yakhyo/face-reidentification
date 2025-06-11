# Utilities API Reference

This document describes the utility functions provided in the `utils/helpers.py` module.

## Image Processing Functions

### norm_crop_image
```python
def norm_crop_image(image, landmark, image_size=112, mode='arcface')
```

Normalizes and crops face images based on landmarks.

#### Parameters
- `image` (numpy.ndarray): Input image
- `landmark` (numpy.ndarray): Face landmarks (5,2)
- `image_size` (int): Output image size
- `mode` (str): Alignment mode

#### Returns
- numpy.ndarray: Aligned and cropped face image

### estimate_norm
```python
def estimate_norm(landmark, image_size=112)
```

Estimates normalization transformation matrix.

#### Parameters
- `landmark` (numpy.ndarray): Face landmarks
- `image_size` (int): Target image size

#### Returns
- tuple: (transformation_matrix, reference_index)

## Geometry Functions

### distance2bbox
```python
def distance2bbox(points, distance, max_shape=None)
```

Converts distance predictions to bounding boxes.

#### Parameters
- `points` (numpy.ndarray): Center points
- `distance` (numpy.ndarray): Distance predictions
- `max_shape` (tuple): Maximum shape constraints

#### Returns
- numpy.ndarray: Bounding boxes [x1, y1, x2, y2]

### distance2kps
```python
def distance2kps(points, distance, max_shape=None)
```

Converts distance predictions to keypoints.

#### Parameters
- `points` (numpy.ndarray): Base points
- `distance` (numpy.ndarray): Distance predictions
- `max_shape` (tuple): Maximum shape constraints

#### Returns
- numpy.ndarray: Keypoint coordinates

## Visualization Functions

### draw_bbox
```python
def draw_bbox(image, bbox, color=(0, 255, 0), thickness=3, proportion=0.2)
```

Draws detection bounding box with corner markers.

#### Parameters
- `image` (numpy.ndarray): Input image
- `bbox` (list/array): Bounding box coordinates
- `color` (tuple): RGB color
- `thickness` (int): Line thickness
- `proportion` (float): Corner size proportion

#### Returns
- numpy.ndarray: Image with drawn bounding box

### draw_bbox_info
```python
def draw_bbox_info(frame, bbox, similarity, name, color)
```

Draws bounding box with identity and similarity information.

#### Parameters
- `frame` (numpy.ndarray): Input frame
- `bbox` (list/array): Bounding box coordinates
- `similarity` (float): Face similarity score
- `name` (str): Identity name
- `color` (tuple): RGB color

## Similarity Computation

### compute_similarity
```python
def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32
```

Computes cosine similarity between face features.

#### Parameters
- `feat1` (numpy.ndarray): First face feature vector
- `feat2` (numpy.ndarray): Second face feature vector

#### Returns
- float: Cosine similarity score

### Example Usage

```python
# Face alignment
aligned_face = norm_crop_image(image, landmarks)

# Bounding box visualization
draw_bbox(frame, bbox, color=(0, 255, 0))

# Add detection info
draw_bbox_info(frame, bbox, 0.95, "Person", (0, 255, 0))

# Compare faces
similarity = compute_similarity(feat1, feat2)
```
