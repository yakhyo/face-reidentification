import cv2
import numpy as np
from skimage.transform import SimilarityTransform


reference_alignment = np.array(
    [[
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ]],
    dtype=np.float32
)


def estimate_norm(landmark, image_size=112):
    """Estimate normalization transformation matrix for facial landmarks.

    Args:
        landmark (ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the output image. Default is 112.

    Returns:
        tuple: A tuple containing:
            - min_matrix (ndarray): The 2x3 transformation matrix for aligning the landmarks.
            - min_index (int): The index of the reference alignment that resulted in the minimum error.
    """
    assert landmark.shape == (5, 2)
    min_matrix = []
    min_index = []
    min_error = float('inf')

    landmark_transform = np.insert(landmark, 2, values=np.ones(5), axis=1)
    transform = SimilarityTransform()

    if image_size == 112:
        alignment = reference_alignment
    else:
        alignment = float(image_size) / 112 * reference_alignment

    for i in np.arange(alignment.shape[0]):
        transform.estimate(landmark, alignment[i])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, landmark_transform.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - alignment[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_matrix = matrix
            min_index = i
    return min_matrix, min_index


def norm_crop_image(image, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bounding boxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """Computing Similarity between two faces.

    Args:
        feat1 (np.ndarray): Face features.
        feat2 (np.ndarray): Face features.

    Returns:
        np.float32: Cosine similarity between face features.
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=3, proportion=0.2):
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)

    return image


def draw_bbox_info(frame, bbox, similarity, name, color):
    x1, y1, x2, y2 = map(int, bbox)

    cv2.putText(
        frame,
        f"{name}: {similarity:.2f}",
        org=(x1, y1-10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=1,
        color=color,
        thickness=1
    )

    # Draw bounding box
    draw_bbox(frame, bbox, color)

    # Draw similarity bar
    rect_x_start = x2 + 10
    rect_x_end = rect_x_start + 10
    rect_y_end = y2
    rect_height = int(similarity * (y2 - y1))
    rect_y_start = rect_y_end - rect_height  # Rectangle starts from bottom and goes upward

    # Draw the filled rectangle
    cv2.rectangle(frame, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, cv2.FILLED)
