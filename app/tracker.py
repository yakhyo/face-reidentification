"""
Multi-object tracking module with ByteTrack integration.

This module provides:
- TrackState: Per-track state management with rolling average embeddings
- FaceTracker: ByteTrack wrapper with IOU fallback
- Track lifecycle management and identity persistence
"""

import time
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """
    Maintains state for a single track across frames.
    
    Attributes:
        id: Unique track identifier
        first_seen_ts: Timestamp when track was first created
        last_seen_ts: Timestamp when track was last updated
        embeddings_deque: Rolling window of recent embeddings
        avg_embedding: L2-normalized average of recent embeddings
        identity: Assigned person name (None if unconfirmed)
        conf_history: History of detection confidences
        sr_used_flag: Whether super-resolution was used
        frame_count: Number of frames this track has been seen
        bbox: Current bounding box [x1, y1, x2, y2]
        last_embedding_frame: Frame index when embedding was last computed
        identity_confirmations: Number of consecutive identity matches
        last_roi_zoom_ts: Timestamp of last ROI zoom operation
    """
    id: int
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    embeddings_deque: deque = field(default_factory=lambda: deque(maxlen=5))
    avg_embedding: Optional[np.ndarray] = None
    identity: Optional[str] = None
    conf_history: List[float] = field(default_factory=list)
    sr_used_flag: bool = False
    frame_count: int = 0
    bbox: Optional[np.ndarray] = None
    last_embedding_frame: int = -1
    identity_confirmations: int = 0
    last_roi_zoom_ts: float = 0.0
    
    def add_embedding(self, embedding: np.ndarray) -> None:
        """
        Add a new embedding and update the rolling average.
        
        Args:
            embedding: Face embedding vector (will be L2-normalized)
        """
        # L2 normalize the embedding
        normalized_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        # Add to deque (automatically maintains max length)
        self.embeddings_deque.append(normalized_emb)
        
        # Recompute average embedding
        if len(self.embeddings_deque) > 0:
            # Stack all embeddings and compute mean
            stacked = np.stack(list(self.embeddings_deque), axis=0)
            mean_emb = np.mean(stacked, axis=0)
            
            # L2 normalize the average
            self.avg_embedding = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        else:
            self.avg_embedding = normalized_emb
    
    def update(self, bbox: np.ndarray, conf: float) -> None:
        """
        Update track with new detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            conf: Detection confidence
        """
        self.bbox = bbox
        self.last_seen_ts = time.time()
        self.conf_history.append(conf)
        self.frame_count += 1
    
    def set_identity(self, name: str, confirmed: bool = False) -> None:
        """
        Set or update track identity.
        
        Args:
            name: Person name
            confirmed: Whether this is a confirmed match
        """
        if self.identity == name:
            self.identity_confirmations += 1
        else:
            self.identity = name
            self.identity_confirmations = 1 if confirmed else 0
    
    def is_identity_confirmed(self, min_confirmations: int = 3) -> bool:
        """Check if identity is confirmed with sufficient confirmations."""
        return self.identity is not None and self.identity_confirmations >= min_confirmations
    
    def get_presence_time(self) -> float:
        """Get total presence time in seconds."""
        return self.last_seen_ts - self.first_seen_ts
    
    def get_avg_confidence(self) -> float:
        """Get average detection confidence."""
        return np.mean(self.conf_history) if self.conf_history else 0.0
    
    def get_face_height(self) -> float:
        """Get face height in pixels."""
        if self.bbox is not None:
            return self.bbox[3] - self.bbox[1]
        return 0.0
    
    def should_compute_embedding(self, frame_idx: int, interval: int) -> bool:
        """
        Check if embedding should be computed for this frame.
        
        Args:
            frame_idx: Current frame index
            interval: Embedding computation interval
            
        Returns:
            True if embedding should be computed
        """
        # Compute on first frame
        if self.last_embedding_frame < 0:
            return True
        
        # Compute every interval frames
        return (frame_idx - self.last_embedding_frame) >= interval
    
    def can_roi_zoom(self, max_frequency_sec: float) -> bool:
        """
        Check if ROI zoom can be performed (throttling).
        
        Args:
            max_frequency_sec: Minimum time between ROI zooms
            
        Returns:
            True if ROI zoom is allowed
        """
        elapsed = time.time() - self.last_roi_zoom_ts
        return elapsed >= max_frequency_sec
    
    def mark_roi_zoom(self) -> None:
        """Mark that ROI zoom was performed."""
        self.last_roi_zoom_ts = time.time()


class IOUTracker:
    """
    Simple IOU-based tracker as fallback for ByteTrack.
    
    Uses Intersection-over-Union (IOU) to match detections across frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initialize IOU tracker.
        
        Args:
            iou_threshold: Minimum IOU for matching
            max_age: Maximum frames to keep track alive without detection
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}
    
    def compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two bboxes.
        
        Args:
            bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
            
        Returns:
            IOU score [0, 1]
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf]
            
        Returns:
            List of (track_id, bbox) tuples
        """
        if len(detections) == 0:
            # Age existing tracks
            to_remove = []
            for track_id, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.tracks[track_id]
            
            return []
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_dets = set()
        results = []
        
        # Try to match each track with best detection
        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                
                iou = self.compute_iou(track['bbox'][:4], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx >= 0:
                # Match found
                bbox = detections[best_idx]
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'age': 0
                }
                matched_tracks.add(track_id)
                matched_dets.add(best_idx)
                results.append((track_id, bbox[:4]))
            else:
                # No match, age track
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': det,
                    'age': 0
                }
                results.append((track_id, det[:4]))
        
        return results


class FaceTracker:
    """
    Face tracker with ByteTrack integration and IOU fallback.
    """
    
    def __init__(self, config: Config):
        """
        Initialize face tracker.
        
        Args:
            config: Global configuration object
        """
        self.config = config
        self.track_states: Dict[int, TrackState] = {}
        
        # Try to import ByteTrack, fallback to IOU tracker
        self.use_bytetrack = False
        try:
            # Try importing ByteTrack from various sources
            try:
                from yolox.tracker.byte_tracker import BYTETracker
                self.tracker = BYTETracker(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=30
                )
                self.use_bytetrack = True
                logger.info("Using ByteTrack for tracking")
            except ImportError:
                raise ImportError("ByteTrack not available")
        except ImportError:
            # Fallback to IOU tracker
            self.tracker = IOUTracker(iou_threshold=0.3, max_age=30)
            self.use_bytetrack = False
            logger.info("Using IOU tracker (ByteTrack not available)")
    
    def update(self, detections: np.ndarray, frame_idx: int) -> List[Tuple[int, np.ndarray, TrackState]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf]
            frame_idx: Current frame index
            
        Returns:
            List of (track_id, bbox, track_state) tuples
        """
        if len(detections) == 0:
            # Clean up inactive tracks
            self._cleanup_inactive_tracks()
            return []
        
        # Update underlying tracker
        if self.use_bytetrack:
            # ByteTrack expects specific format
            online_targets = self.tracker.update(detections, [detections.shape[0], detections.shape[0]])
            track_results = [(int(t.track_id), t.tlbr) for t in online_targets]
        else:
            # IOU tracker
            track_results = self.tracker.update(detections)
        
        # Update track states
        results = []
        for track_id, bbox in track_results:
            # Find matching detection for confidence
            conf = 0.5  # default
            for det in detections:
                det_bbox = det[:4]
                iou = self._compute_iou(bbox, det_bbox)
                if iou > 0.7:
                    conf = det[4]
                    break
            
            # Get or create track state
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(id=track_id)
                logger.info(f"TRACK created id={track_id}")
            
            track_state = self.track_states[track_id]
            track_state.update(bbox, conf)
            
            results.append((track_id, bbox, track_state))
        
        # Clean up inactive tracks
        self._cleanup_inactive_tracks()
        
        return results
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IOU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _cleanup_inactive_tracks(self) -> None:
        """Remove tracks that have been inactive for too long."""
        current_time = time.time()
        to_remove = []
        
        for track_id, track_state in self.track_states.items():
            elapsed = current_time - track_state.last_seen_ts
            if elapsed > self.config.tracking.TRACK_INACTIVE_TTL:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            logger.info(f"TRACK removed id={track_id}")
            del self.track_states[track_id]
    
    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        """Get track state by ID."""
        return self.track_states.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, TrackState]:
        """Get all active track states."""
        return self.track_states


if __name__ == "__main__":
    # Test TrackState
    print("Testing TrackState...")
    track = TrackState(id=1)
    
    # Add some embeddings
    for i in range(10):
        emb = np.random.randn(512)
        track.add_embedding(emb)
        print(f"Added embedding {i+1}, avg_embedding shape: {track.avg_embedding.shape}")
    
    print(f"\nTrack info:")
    print(f"  ID: {track.id}")
    print(f"  Frame count: {track.frame_count}")
    print(f"  Embeddings in deque: {len(track.embeddings_deque)}")
    print(f"  Average embedding norm: {np.linalg.norm(track.avg_embedding):.4f}")
    
    # Test IOU tracker
    print("\n\nTesting IOU Tracker...")
    tracker = IOUTracker()
    
    # Simulate detections across frames
    for frame_idx in range(5):
        # Simulate 2 moving faces
        dets = np.array([
            [100 + frame_idx*10, 100, 200 + frame_idx*10, 200, 0.9],
            [300, 150 + frame_idx*5, 400, 250 + frame_idx*5, 0.85]
        ])
        
        results = tracker.update(dets)
        print(f"\nFrame {frame_idx}: {len(results)} tracks")
        for track_id, bbox in results:
            print(f"  Track {track_id}: bbox={bbox[:4]}")
