"""
ROI zoom module for enhancing small/low-confidence face detections.

This module provides:
- ROI extraction and padding
- ROI upscaling
- Detection on upscaled ROI
- Coordinate mapping back to original frame
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional

from app.config import Config

logger = logging.getLogger(__name__)


def compute_padded_roi(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
    padding: float = 1.5
) -> Tuple[int, int, int, int]:
    """
    Compute padded ROI from bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_shape: (height, width) of the image
        padding: Padding factor (1.5 = 50% padding)
        
    Returns:
        Tuple (x1, y1, x2, y2) of padded ROI, clipped to image bounds
    """
    height, width = image_shape
    
    x1, y1, x2, y2 = bbox[:4]
    
    # Compute bbox center and dimensions
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    # Apply padding
    new_w = w * padding
    new_h = h * padding
    
    # Compute new bounds
    roi_x1 = int(cx - new_w / 2)
    roi_y1 = int(cy - new_h / 2)
    roi_x2 = int(cx + new_w / 2)
    roi_y2 = int(cy + new_h / 2)
    
    # Clip to image bounds
    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)
    roi_x2 = min(width, roi_x2)
    roi_y2 = min(height, roi_y2)
    
    return roi_x1, roi_y1, roi_x2, roi_y2


def roi_zoom_detect(
    image: np.ndarray,
    bbox: np.ndarray,
    detector,
    config: Config,
    debug_output_path: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Perform ROI zoom detection on a small/uncertain face detection.
    
    Args:
        image: Full frame image
        bbox: Original detection [x1, y1, x2, y2, conf]
        detector: SCRFD detector instance
        config: Configuration object
        debug_output_path: Optional path to save ROI crop
        
    Returns:
        Tuple of (improved_bbox, kpss, was_improved):
            - improved_bbox: Updated bbox if detection improved, None otherwise
            - kpss: Keypoints if detection improved
            - was_improved: True if ROI zoom provided better detection
    """
    height, width = image.shape[:2]
    
    # Compute padded ROI
    roi_x1, roi_y1, roi_x2, roi_y2 = compute_padded_roi(
        bbox,
        (height, width),
        padding=config.roi_zoom.ROI_PADDING
    )
    
    # Extract ROI
    roi_img = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    
    if roi_img.size == 0:
        return None, None, False
    
    # Upscale ROI
    target_size = config.roi_zoom.ROI_SCALED_SIZE
    roi_upscaled = cv2.resize(roi_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Compute scale factors for mapping back
    scale_x = target_size[0] / (roi_x2 - roi_x1)
    scale_y = target_size[1] / (roi_y2 - roi_y1)
    
    try:
        # Run detection on upscaled ROI
        roi_bboxes, roi_kpss = detector.detect(roi_upscaled, max_num=1)
        
        if len(roi_bboxes) == 0:
            # No detection in ROI
            return None, None, False
        
        # Get best detection
        roi_bbox = roi_bboxes[0]
        roi_kps = roi_kpss[0] if roi_kpss is not None and len(roi_kpss) > 0 else None
        
        # Map detection back to original frame coordinates
        # Scale back from upscaled ROI to original ROI size
        mapped_x1 = roi_bbox[0] / scale_x + roi_x1
        mapped_y1 = roi_bbox[1] / scale_y + roi_y1
        mapped_x2 = roi_bbox[2] / scale_x + roi_x1
        mapped_y2 = roi_bbox[3] / scale_y + roi_y1
        mapped_conf = roi_bbox[4]
        
        # Map keypoints back
        mapped_kps = None
        if roi_kps is not None:
            mapped_kps = roi_kps.copy()
            mapped_kps[:, 0] = mapped_kps[:, 0] / scale_x + roi_x1
            mapped_kps[:, 1] = mapped_kps[:, 1] / scale_y + roi_y1
        
        # Create improved bbox
        improved_bbox = np.array([mapped_x1, mapped_y1, mapped_x2, mapped_y2, mapped_conf])
        
        # Check if improvement occurred
        original_conf = bbox[4]
        original_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        improved_area = (improved_bbox[2] - improved_bbox[0]) * (improved_bbox[3] - improved_bbox[1])
        
        # Consider improved if confidence increased OR area increased significantly
        is_improved = (mapped_conf > original_conf * 1.1) or (improved_area > original_area * 1.2)
        
        if is_improved:
            logger.debug(
                f"ROI_ZOOM improved detection: "
                f"conf {original_conf:.3f}->{mapped_conf:.3f}, "
                f"area {original_area:.0f}->{improved_area:.0f}"
            )
        
        # Save debug crop
        if debug_output_path and config.debug.DEBUG_ENABLED and is_improved:
            try:
                cv2.imwrite(debug_output_path, roi_upscaled)
                logger.debug(f"Saved ROI crop to {debug_output_path}")
            except Exception as e:
                logger.warning(f"Failed to save ROI crop: {e}")
        
        return improved_bbox if is_improved else None, mapped_kps if is_improved else None, is_improved
    
    except Exception as e:
        logger.warning(f"Error in ROI zoom detection: {e}")
        return None, None, False


def should_apply_roi_zoom(
    bbox: np.ndarray,
    config: Config
) -> bool:
    """
    Check if ROI zoom should be applied to a detection.
    
    Args:
        bbox: Detection [x1, y1, x2, y2, conf]
        config: Configuration object
        
    Returns:
        True if ROI zoom should be applied
    """
    # Check face height
    face_height = bbox[3] - bbox[1]
    if face_height < config.roi_zoom.SMALL_FACE_THRESHOLD:
        return True
    
    # Check confidence
    conf = bbox[4]
    if conf < config.roi_zoom.LOW_CONFIDENCE_THRESHOLD:
        return True
    
    return False


class ROIZoomManager:
    """
    Manages ROI zoom operations with throttling per track.
    """
    
    def __init__(self, config: Config):
        """
        Initialize ROI zoom manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.active_roi_operations = 0
        self.max_concurrent = config.performance.MAX_CONCURRENT_ROI
    
    def can_process(self) -> bool:
        """
        Check if ROI zoom can be processed (concurrent limit).
        
        Returns:
            True if processing is allowed
        """
        return self.active_roi_operations < self.max_concurrent
    
    def start_operation(self):
        """Mark start of ROI zoom operation."""
        self.active_roi_operations += 1
    
    def end_operation(self):
        """Mark end of ROI zoom operation."""
        self.active_roi_operations = max(0, self.active_roi_operations - 1)
    
    def process_detection(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        track_state,
        detector,
        track_id: int,
        frame_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a detection with ROI zoom if applicable.
        
        Args:
            image: Full frame image
            bbox: Detection bbox
            track_state: TrackState object
            detector: SCRFD detector
            track_id: Track ID for logging
            frame_idx: Current frame index
            
        Returns:
            Tuple of (improved_bbox, improved_kpss) or (None, None)
        """
        # Check if should apply ROI zoom
        if not should_apply_roi_zoom(bbox, self.config):
            return None, None
        
        # Check track throttling
        if not track_state.can_roi_zoom(self.config.roi_zoom.MAX_ROI_FREQUENCY_SEC):
            return None, None
        
        # Check concurrent limit
        if not self.can_process():
            return None, None
        
        # Process
        self.start_operation()
        try:
            debug_path = None
            if self.config.debug.DEBUG_ENABLED:
                import os
                debug_path = os.path.join(
                    self.config.debug.DEBUG_CROPS_DIR,
                    f"roi_track{track_id}_frame{frame_idx}.jpg"
                )
            
            improved_bbox, improved_kpss, was_improved = roi_zoom_detect(
                image, bbox, detector, self.config, debug_path
            )
            
            if was_improved:
                logger.info(f"ROI_ZOOM for track {track_id}")
                track_state.mark_roi_zoom()
                return improved_bbox, improved_kpss
            
            return None, None
        
        finally:
            self.end_operation()


if __name__ == "__main__":
    # Test ROI computation
    print("Testing ROI computation...")
    
    # Test case 1: Center face
    bbox = np.array([100, 100, 200, 200, 0.9])
    image_shape = (480, 640)
    
    roi = compute_padded_roi(bbox, image_shape, padding=1.5)
    print(f"\nOriginal bbox: {bbox[:4]}")
    print(f"Padded ROI: {roi}")
    print(f"ROI size: {roi[2]-roi[0]}x{roi[3]-roi[1]}")
    
    # Test case 2: Edge face (should clip)
    bbox = np.array([10, 10, 60, 60, 0.8])
    roi = compute_padded_roi(bbox, image_shape, padding=1.5)
    print(f"\nEdge bbox: {bbox[:4]}")
    print(f"Padded ROI (clipped): {roi}")
    
    # Test case 3: Check if should apply ROI zoom
    from app.config import Config
    config = Config()
    
    # Small face - should zoom
    small_bbox = np.array([100, 100, 150, 150, 0.9])  # 50px height
    print(f"\nSmall face (50px): should_zoom = {should_apply_roi_zoom(small_bbox, config)}")
    
    # Large face - should not zoom
    large_bbox = np.array([100, 100, 200, 250, 0.9])  # 150px height
    print(f"Large face (150px): should_zoom = {should_apply_roi_zoom(large_bbox, config)}")
    
    # Low confidence - should zoom
    low_conf_bbox = np.array([100, 100, 200, 250, 0.4])  # conf=0.4
    print(f"Low confidence (0.4): should_zoom = {should_apply_roi_zoom(low_conf_bbox, config)}")
