"""
Tiled detection module for detecting small/distant faces.

This module provides:
- Tile generation with overlap
- Tiled detection runner
- NMS merge for combining tile detections
"""

import time
import logging
import numpy as np
import cv2
from typing import List, Tuple

from app.config import Config

logger = logging.getLogger(__name__)


def generate_tiles(
    image_shape: Tuple[int, int],
    tile_rows: int = 3,
    tile_cols: int = 3,
    overlap: float = 0.2
) -> List[Tuple[int, int, int, int]]:
    """
    Generate overlapping tiles for an image.
    
    Args:
        image_shape: (height, width) of the image
        tile_rows: Number of rows in the tile grid
        tile_cols: Number of columns in the tile grid
        overlap: Overlap fraction between adjacent tiles (0-1)
        
    Returns:
        List of tile bounding boxes (x1, y1, x2, y2) in image coordinates
    """
    height, width = image_shape
    tiles = []
    
    # Calculate tile dimensions with overlap
    tile_height = int(height / tile_rows * (1 + overlap))
    tile_width = int(width / tile_cols * (1 + overlap))
    
    # Calculate step size (distance between tile origins)
    step_height = int(height / tile_rows)
    step_width = int(width / tile_cols)
    
    for row in range(tile_rows):
        for col in range(tile_cols):
            # Calculate tile position
            y1 = row * step_height
            x1 = col * step_width
            y2 = min(y1 + tile_height, height)
            x2 = min(x1 + tile_width, width)
            
            # Ensure tile has minimum size
            if (x2 - x1) >= 100 and (y2 - y1) >= 100:
                tiles.append((x1, y1, x2, y2))
    
    return tiles


def run_tiled_detection(
    image: np.ndarray,
    detector,
    config: Config,
    debug_output_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run detection on image tiles and merge results.
    
    Args:
        image: Input image
        detector: SCRFD detector instance
        config: Configuration object
        debug_output_path: Optional path to save tile visualization
        
    Returns:
        Tuple of (merged_bboxes, merged_kpss) after NMS
    """
    height, width = image.shape[:2]
    
    # Generate tiles
    tiles = generate_tiles(
        (height, width),
        tile_rows=config.tiling.TILE_ROWS,
        tile_cols=config.tiling.TILE_COLS,
        overlap=config.tiling.TILE_OVERLAP
    )
    
    logger.info(f"TILE_DET triggered: {len(tiles)} tiles generated")
    
    # Collect all detections from tiles
    all_bboxes = []
    all_kpss = []
    
    for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
        # Extract tile
        tile_img = image[y1:y2, x1:x2].copy()
        
        try:
            # Run detection on tile
            bboxes, kpss = detector.detect(tile_img, max_num=0)
            
            if len(bboxes) > 0:
                # Map detections back to full image coordinates
                bboxes[:, 0] += x1  # x1
                bboxes[:, 1] += y1  # y1
                bboxes[:, 2] += x1  # x2
                bboxes[:, 3] += y1  # y2
                
                if kpss is not None and len(kpss) > 0:
                    kpss[:, :, 0] += x1  # keypoint x coordinates
                    kpss[:, :, 1] += y1  # keypoint y coordinates
                
                all_bboxes.append(bboxes)
                all_kpss.append(kpss)
                
                logger.debug(f"Tile {tile_idx} ({x1},{y1},{x2},{y2}): {len(bboxes)} detections")
        
        except Exception as e:
            logger.warning(f"Error processing tile {tile_idx}: {e}")
            continue
    
    # Merge detections
    if len(all_bboxes) == 0:
        logger.info("TILE_DET: No detections found in any tile")
        return np.array([]), np.array([])
    
    # Concatenate all detections
    merged_bboxes = np.vstack(all_bboxes)
    merged_kpss = np.vstack(all_kpss) if all_kpss and all_kpss[0] is not None else None
    
    logger.info(f"TILE_DET: {len(merged_bboxes)} detections before NMS")
    
    # Apply NMS to remove duplicates
    final_bboxes, final_kpss = nms_merge(
        merged_bboxes,
        merged_kpss,
        iou_threshold=config.tiling.TILE_NMS_THRESHOLD
    )
    
    logger.info(f"TILE_DET: {len(final_bboxes)} detections after NMS")
    
    # Save debug visualization
    if debug_output_path and config.debug.DEBUG_ENABLED:
        save_tile_visualization(image, tiles, final_bboxes, debug_output_path)
    
    return final_bboxes, final_kpss


def nms_merge(
    bboxes: np.ndarray,
    kpss: np.ndarray,
    iou_threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Suppression to merge overlapping detections.
    
    Args:
        bboxes: Detections [x1, y1, x2, y2, conf]
        kpss: Keypoints for each detection
        iou_threshold: IOU threshold for suppression
        
    Returns:
        Tuple of (filtered_bboxes, filtered_kpss)
    """
    if len(bboxes) == 0:
        return bboxes, kpss
    
    # Extract coordinates and scores
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    
    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IOU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IOU below threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    # Filter bboxes and kpss
    filtered_bboxes = bboxes[keep]
    filtered_kpss = kpss[keep] if kpss is not None and len(kpss) > 0 else None
    
    return filtered_bboxes, filtered_kpss


def save_tile_visualization(
    image: np.ndarray,
    tiles: List[Tuple[int, int, int, int]],
    detections: np.ndarray,
    output_path: str
) -> None:
    """
    Save visualization of tiles and detections.
    
    Args:
        image: Original image
        tiles: List of tile bounding boxes
        detections: Final detections after NMS
        output_path: Path to save visualization
    """
    try:
        vis_img = image.copy()
        
        # Draw tiles in blue
        for x1, y1, x2, y2 in tiles:
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw detections in green
        for det in detections:
            x1, y1, x2, y2 = det[:4].astype(int)
            conf = det[4]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Save
        cv2.imwrite(output_path, vis_img)
        logger.debug(f"Saved tile visualization to {output_path}")
    
    except Exception as e:
        logger.warning(f"Failed to save tile visualization: {e}")


class TiledDetectionManager:
    """
    Manages tiled detection with throttling and trigger logic.
    """
    
    def __init__(self, config: Config):
        """
        Initialize tiled detection manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.last_tile_time = 0.0
        self.no_detection_count = 0
        self.small_face_count = 0
    
    def should_trigger_tiling(
        self,
        num_detections: int,
        avg_face_height: float
    ) -> bool:
        """
        Check if tiled detection should be triggered.
        
        Args:
            num_detections: Number of detections in current frame
            avg_face_height: Average face height in pixels
            
        Returns:
            True if tiling should be triggered
        """
        # Check throttling
        elapsed = time.time() - self.last_tile_time
        if elapsed < self.config.tiling.MAX_TILE_FREQUENCY_SEC:
            return False
        
        # Check no detection trigger
        if num_detections == 0:
            self.no_detection_count += 1
            if self.no_detection_count >= self.config.tiling.TILE_TRIGGER_NO_DETS:
                logger.info(f"Tiling trigger: {self.no_detection_count} frames with no detections")
                self.no_detection_count = 0
                self.last_tile_time = time.time()
                return True
        else:
            self.no_detection_count = 0
        
        # Check small face trigger
        if avg_face_height > 0 and avg_face_height < self.config.roi_zoom.SMALL_FACE_THRESHOLD:
            self.small_face_count += 1
            if self.small_face_count >= self.config.tiling.TILE_TRIGGER_SMALL_FACES:
                logger.info(f"Tiling trigger: {self.small_face_count} frames with small faces (avg={avg_face_height:.1f}px)")
                self.small_face_count = 0
                self.last_tile_time = time.time()
                return True
        else:
            self.small_face_count = 0
        
        return False
    
    def reset(self):
        """Reset trigger counters."""
        self.no_detection_count = 0
        self.small_face_count = 0


if __name__ == "__main__":
    # Test tile generation
    print("Testing tile generation...")
    
    # Standard HD resolution
    tiles = generate_tiles((1080, 1920), tile_rows=3, tile_cols=3, overlap=0.2)
    print(f"\nGenerated {len(tiles)} tiles for 1920x1080 image:")
    for i, (x1, y1, x2, y2) in enumerate(tiles):
        w, h = x2 - x1, y2 - y1
        print(f"  Tile {i}: ({x1},{y1}) to ({x2},{y2}) - size {w}x{h}")
    
    # Test NMS
    print("\n\nTesting NMS merge...")
    # Create overlapping detections
    bboxes = np.array([
        [100, 100, 200, 200, 0.9],
        [105, 105, 205, 205, 0.85],  # Overlaps with first
        [300, 300, 400, 400, 0.95],
        [310, 310, 410, 410, 0.80],  # Overlaps with third
    ])
    
    print(f"Input: {len(bboxes)} detections")
    filtered_bboxes, _ = nms_merge(bboxes, None, iou_threshold=0.45)
    print(f"After NMS: {len(filtered_bboxes)} detections")
    print("Remaining detections:")
    for det in filtered_bboxes:
        print(f"  {det}")
