"""
Configuration module for face recognition with tracking.

This module contains all configurable parameters for:
- Tracking behavior
- Tiling detection
- ROI zoom
- Embedding computation
- Attendance marking
- Privacy and storage
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrackingConfig:
    """Configuration for multi-object tracking."""
    
    # Embedding computation frequency
    EMBED_INTERVAL: int = 3  # Compute embedding every N frames
    
    # Rolling average window for embeddings
    EMBED_ROLLING_WINDOW: int = 5  # Keep last N embeddings
    
    # Track lifecycle
    TRACK_INACTIVE_TTL: float = 3.0  # seconds before removing inactive track
    
    # Identity confirmation
    MIN_IDENTITY_CONFIRMATIONS: int = 3  # Consecutive matches before confirming identity


@dataclass
class TilingConfig:
    """Configuration for tiled detection."""
    
    # Tile grid dimensions
    TILE_ROWS: int = 3
    TILE_COLS: int = 3
    TILE_OVERLAP: float = 0.2  # 20% overlap between tiles
    
    # Trigger conditions
    TILE_TRIGGER_NO_DETS: int = 5  # frames with no detections before tiling
    TILE_TRIGGER_SMALL_FACES: int = 5  # frames with small faces before tiling
    
    # Throttling
    MAX_TILE_FREQUENCY_SEC: float = 5.0  # Don't tile more often than this
    
    # NMS for merging tile detections
    TILE_NMS_THRESHOLD: float = 0.45


@dataclass
class ROIZoomConfig:
    """Configuration for ROI zoom operations."""
    
    # Face size threshold
    SMALL_FACE_THRESHOLD: int = 56  # pixels (height)
    
    # Low confidence threshold
    LOW_CONFIDENCE_THRESHOLD: float = 0.5
    
    # ROI expansion and scaling
    ROI_PADDING: float = 1.5  # Expand bbox by this factor
    ROI_SCALED_SIZE: Tuple[int, int] = (512, 512)  # Target size for upscaling
    
    # Throttling
    MAX_ROI_FREQUENCY_SEC: float = 5.0  # Per track


@dataclass
class AttendanceConfig:
    """Configuration for attendance marking."""
    
    # Presence time threshold
    PRESENCE_SECONDS: int = 60  # Mark attendance after this many seconds
    
    # Similarity threshold for matching
    SIMILARITY_THRESHOLD: float = 0.60
    
    # Database paths
    ATTENDANCE_CSV: str = "./attendance.csv"
    ATTENDANCE_DB: str = "./attendance.sqlite"
    
    # Session configuration
    CAMERA_ID: str = "default_camera"
    SESSION_ID: str = "default_session"


@dataclass
class PrivacyConfig:
    """Configuration for privacy and data storage."""
    
    # Snapshot storage
    STORE_SNAPSHOTS: bool = False  # Don't store by default
    SNAPSHOT_MAX_SIZE_KB: int = 64  # Maximum size per snapshot
    SNAPSHOT_DIR: str = "./snapshots"
    
    # Encryption (future feature)
    ENCRYPT_SNAPSHOTS: bool = False


@dataclass
class DebugConfig:
    """Configuration for debug output."""
    
    # Debug output
    DEBUG_ENABLED: bool = False
    DEBUG_DIR: str = "./debug"
    
    # Save debug crops
    SAVE_DEBUG_CROPS: bool = True
    DEBUG_CROPS_DIR: str = "./debug/crops"
    
    # Save tile overviews
    SAVE_TILE_OVERVIEW: bool = True
    DEBUG_TILE_DIR: str = "./debug/tile_overview"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Batch processing
    EMBEDDING_BATCH_SIZE: int = 16  # Maximum batch size for GPU
    
    # Concurrent operations
    MAX_CONCURRENT_TILES: int = 1  # Tiles running at once
    MAX_CONCURRENT_ROI: int = 2  # ROI zooms running at once
    
    # GPU configuration
    USE_GPU: bool = True  # Try to use GPU if available
    GPU_DEVICE_ID: int = 0


class Config:
    """Main configuration class combining all config sections."""
    
    def __init__(self):
        self.tracking = TrackingConfig()
        self.tiling = TilingConfig()
        self.roi_zoom = ROIZoomConfig()
        self.attendance = AttendanceConfig()
        self.privacy = PrivacyConfig()
        self.debug = DebugConfig()
        self.performance = PerformanceConfig()
    
    def enable_debug(self, debug_dir: str = "./debug"):
        """Enable debug mode with output directory."""
        self.debug.DEBUG_ENABLED = True
        self.debug.DEBUG_DIR = debug_dir
        self.debug.DEBUG_CROPS_DIR = os.path.join(debug_dir, "crops")
        self.debug.DEBUG_TILE_DIR = os.path.join(debug_dir, "tile_overview")
        
        # Create directories
        os.makedirs(self.debug.DEBUG_CROPS_DIR, exist_ok=True)
        os.makedirs(self.debug.DEBUG_TILE_DIR, exist_ok=True)
    
    def set_camera_session(self, camera_id: str, session_id: str):
        """Set camera and session IDs for attendance tracking."""
        self.attendance.CAMERA_ID = camera_id
        self.attendance.SESSION_ID = session_id
    
    def enable_snapshots(self, snapshot_dir: str = "./snapshots"):
        """Enable snapshot storage."""
        self.privacy.STORE_SNAPSHOTS = True
        self.privacy.SNAPSHOT_DIR = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary for logging/serialization."""
        return {
            "tracking": {
                "embed_interval": self.tracking.EMBED_INTERVAL,
                "embed_rolling_window": self.tracking.EMBED_ROLLING_WINDOW,
                "track_inactive_ttl": self.tracking.TRACK_INACTIVE_TTL,
                "min_identity_confirmations": self.tracking.MIN_IDENTITY_CONFIRMATIONS,
            },
            "tiling": {
                "tile_rows": self.tiling.TILE_ROWS,
                "tile_cols": self.tiling.TILE_COLS,
                "tile_overlap": self.tiling.TILE_OVERLAP,
                "tile_trigger_no_dets": self.tiling.TILE_TRIGGER_NO_DETS,
                "max_tile_frequency_sec": self.tiling.MAX_TILE_FREQUENCY_SEC,
            },
            "roi_zoom": {
                "small_face_threshold": self.roi_zoom.SMALL_FACE_THRESHOLD,
                "low_confidence_threshold": self.roi_zoom.LOW_CONFIDENCE_THRESHOLD,
                "roi_padding": self.roi_zoom.ROI_PADDING,
                "max_roi_frequency_sec": self.roi_zoom.MAX_ROI_FREQUENCY_SEC,
            },
            "attendance": {
                "presence_seconds": self.attendance.PRESENCE_SECONDS,
                "similarity_threshold": self.attendance.SIMILARITY_THRESHOLD,
            },
            "performance": {
                "embedding_batch_size": self.performance.EMBEDDING_BATCH_SIZE,
                "use_gpu": self.performance.USE_GPU,
            },
        }


# Default global config instance
default_config = Config()


if __name__ == "__main__":
    # Example usage
    config = Config()
    print("Default Configuration:")
    print("-" * 50)
    
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    print("\nEnabling debug mode...")
    config.enable_debug("./test_debug")
    print(f"Debug enabled: {config.debug.DEBUG_ENABLED}")
    print(f"Debug crops dir: {config.debug.DEBUG_CROPS_DIR}")
