# Updates and Progress Tracking

**Project:** Face Recognition with Multi-Object Tracking Enhancement  
**Started:** November 3, 2025  
**Package Manager:** uv (using pyproject.toml)  
**Target Hardware:** GTX 1650  

---

## Current Codebase Understanding (Baseline)

### ✅ Existing Architecture
1. **Detection Model**: SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
   - Models: det_500m.onnx (2.41 MB), det_2.5g.onnx (3.14 MB), det_10g.onnx (16.1 MB)
   - Input size: 640x640
   - Features: Face detection with keypoints (5-point facial landmarks)

2. **Recognition Model**: ArcFace (Additive Angular Margin Loss)
   - Models: w600k_mbf.onnx (12.99 MB - MobileFace), w600k_r50.onnx (166 MB - ResNet-50)
   - Embedding size: 512 dimensions
   - Features: L2-normalized embeddings

3. **Database**: FAISS (Facebook AI Similarity Search)
   - IndexFlatIP for cosine similarity
   - Thread-safe operations with RLock
   - Batch search optimization (sequential for <10 faces, parallel for ≥10)
   - Persistent storage (faiss_index.bin + metadata.json)

4. **Current Pipeline**:
   - Frame → Detection → Face Alignment → Embedding Extraction → FAISS Search → Recognition
   - Video processing with OpenCV
   - Real-time display with bounding boxes and names
   - Output video recording

### 📋 Current Limitations
1. **No Tracking**: Each frame processed independently - no temporal consistency
2. **ID Switches**: Same person can get different IDs across frames
3. **No Attendance Logic**: No time-based presence tracking
4. **Small Face Issues**: Poor detection for distant/small faces (< 56px)
5. **No Tiling**: Full-frame detection only - misses tiny faces
6. **No ROI Zoom**: No targeted upscaling for small detections
7. **Redundant Embeddings**: Computes embedding for every detection every frame

---

## Enhancement Goals

### 🎯 Primary Objectives
1. **Stable Tracking**: ByteTrack integration for persistent track_ids
2. **Sparse Embeddings**: Compute embeddings every N frames per track (not every detection)
3. **Rolling Average**: Maintain per-track average embedding for better matching
4. **Tiled Detection**: Fallback for small/distant faces
5. **ROI Zoom**: Targeted upscaling for low-confidence small faces
6. **Attendance System**: Time-based presence tracking (≥60s default)
7. **Performance**: 5-10 FPS on GTX 1650 with up to 10 faces

### 📊 Success Metrics
- Track IDs stable: ≤1 ID switch/person/minute
- Long-distance recall: +30% improvement with tiling+ROI
- Attendance marking: Uses presence time rules
- System stability: Runs on GTX 1650 without crashes

---

## Implementation Plan

### Phase 1: Configuration & Core Tracking ✅
- [✅] Create `app/config.py` with all thresholds
- [✅] Create `app/tracker.py` with ByteTrack wrapper + TrackState
- [✅] Implement IOU tracker fallback
- [✅] Add track ID persistence and state management

### Phase 2: Tiling & ROI Zoom ✅
- [✅] Create `app/tiling.py` for tile generation and NMS merge
- [✅] Create `app/roi_zoom.py` for targeted upscaling
- [✅] Implement trigger logic (no detections, small faces)
- [✅] Add throttling (MAX_TILE_FREQUENCY_SEC)

### Phase 3: Main Loop Integration ⏳
- [ ] Modify `main.py` to integrate tracker
- [ ] Add tiled detection fallback
- [ ] Add ROI zoom for small/uncertain detections
- [ ] Implement sparse embedding computation (EMBED_INTERVAL)
- [ ] Add rolling average embedding per track
- [ ] Implement persistence rules for identity assignment

### Phase 4: Attendance System ⏳
- [ ] Create attendance.csv and attendance.sqlite schemas
- [ ] Implement presence time tracking
- [ ] Add attendance marking logic (≥PRESENCE_SECONDS)
- [ ] Add event logging (student_id, timestamps, confidence)

### Phase 5: Logging & Debug ⏳
- [ ] Enhanced logging for all key events
- [ ] Debug output (crops, tile overviews)
- [ ] Snapshot storage with privacy controls
- [ ] Performance metrics logging

### Phase 6: Testing & Validation ⏳
- [ ] Create `tools/eval_basic.py` for metrics
- [ ] Smoke test with webcam
- [ ] Tiled detection test with vid_far.mp4
- [ ] ROI zoom test
- [ ] Attendance marking test
- [ ] Metrics computation (recall, precision, ID switches)

### Phase 7: Documentation & Cleanup ⏳
- [ ] Update README.md with new flags
- [ ] Add unit tests for tiling, ROI zoom
- [ ] Update requirements.txt / pyproject.toml
- [ ] Final cleanup and code review

---

## Progress Log

### Session 1: November 3, 2025 - Part 1
**Status**: Codebase analysis complete

**What Went Right:**
- ✅ Successfully analyzed entire codebase
- ✅ Identified all key components and their interactions
- ✅ Understood FAISS database implementation
- ✅ Mapped current pipeline flow
- ✅ Identified enhancement opportunities

**What Went Wrong:**
- N/A (initial session)

**Next Steps:**
1. Create `app/` directory structure ✅
2. Implement `app/config.py` with all configuration defaults ✅
3. Begin `app/tracker.py` with ByteTrack integration ✅

**Notes:**
- Project uses `uv` package manager (pyproject.toml)
- ONNX models already downloaded (weights/ directory exists)
- FAISS database has good thread-safe implementation
- Current batch optimization is smart (sequential for <10 faces)

---

### Session 1: November 3, 2025 - Part 2
**Status**: Core modules implemented ✅

**What Went Right:**
- ✅ Created `app/` directory structure with `__init__.py`
- ✅ Implemented `app/config.py` with dataclass-based configuration
  - Organized into 7 config sections: Tracking, Tiling, ROI Zoom, Attendance, Privacy, Debug, Performance
  - All 13+ configuration parameters properly organized
  - Helper methods for enabling debug mode and snapshot storage
  - Config serialization to dict for logging
- ✅ Implemented `app/tracker.py` with complete tracking system
  - `TrackState` class with all required fields and methods
  - L2-normalized rolling average embeddings (deque with maxlen)
  - `add_embedding()` maintains normalized average correctly
  - Identity confirmation logic with persistence tracking
  - ByteTrack integration with graceful fallback to IOU tracker
  - `IOUTracker` fully implemented as fallback
  - `FaceTracker` wrapper managing track lifecycle
  - Automatic cleanup of inactive tracks based on TTL
- ✅ Implemented `app/tiling.py` for small face detection
  - `generate_tiles()` with configurable overlap
  - `run_tiled_detection()` with coordinate mapping
  - NMS merge with configurable threshold
  - `TiledDetectionManager` with throttling and triggers
  - Debug visualization output
  - Proper logging of tile operations
- ✅ Implemented `app/roi_zoom.py` for upscaling small faces
  - `compute_padded_roi()` with configurable padding
  - `roi_zoom_detect()` with coordinate mapping back to frame
  - Detection improvement logic (conf or area increase)
  - `ROIZoomManager` with concurrent operation limits
  - Per-track throttling support
  - Debug crop saving

**What Went Wrong:**
- N/A - All implementations successful

**Technical Decisions Made:**
1. **Config Organization**: Used dataclasses for type safety and clarity
2. **Tracker Fallback**: Implemented full IOU tracker instead of depending on motpy
3. **Coordinate Mapping**: Careful scale factor calculation for tile and ROI detections
4. **Throttling**: Time-based for tiling, per-track for ROI zoom
5. **Embedding Normalization**: L2 normalization before and after averaging

**Next Steps:**
1. Modify `main.py` to integrate all new modules
2. Add attendance tracking (CSV + SQLite)
3. Enhance logging for all key events
4. Create evaluation script
5. Test end-to-end system

**Notes:**
- All core modules are self-contained and testable
- Each module has `__main__` test code for development
- Ready for integration into main.py
- ByteTrack dependency is optional (graceful fallback works)

---

## Technical Decisions

### Tracker Choice
- **Primary**: ByteTrack (if available via pip)
- **Fallback**: Simple IOU tracker (custom implementation)
- **Rationale**: ByteTrack is SOTA for multi-object tracking with excellent ID stability

### Embedding Strategy
- **Frequency**: Every EMBED_INTERVAL frames (default: 3)
- **Aggregation**: Rolling average of last 5 embeddings
- **Normalization**: L2-normalized before averaging
- **Rationale**: Reduces compute while improving robustness via temporal averaging

### Tiling Strategy
- **Trigger**: 0 detections for 5 frames OR avg face height < 56px
- **Grid**: 3x3 with 20% overlap
- **Throttle**: Max once per 5 seconds
- **Rationale**: Balances recall improvement vs compute cost

### ROI Zoom Strategy
- **Trigger**: Face height < 56px OR detection confidence < 0.5
- **Padding**: 1.5x bbox size
- **Target Size**: 512x512
- **Throttle**: Max once per track per 5 seconds
- **Rationale**: Targeted compute for uncertain detections only

---

## Dependencies to Add

### Required
- [ ] ByteTrack (or motpy as fallback) - already in pyproject.toml ✅
- [ ] sqlite3 (standard library) ✅
- [ ] collections.deque (standard library) ✅

### Optional
- [ ] Real-ESRGAN (for super-resolution, disabled by default)

---

## Configuration Defaults

```python
# Tracking
EMBED_INTERVAL = 3
EMBED_ROLLING_WINDOW = 5
TRACK_INACTIVE_TTL = 3.0  # seconds

# Tiling
TILE_ROWS = 3
TILE_COLS = 3
TILE_OVERLAP = 0.2
TILE_TRIGGER_NO_DETS = 5  # frames
MAX_TILE_FREQUENCY_SEC = 5

# ROI Zoom
SMALL_FACE_THRESHOLD = 56  # px height
ROI_PADDING = 1.5
ROI_SCALED_SIZE = (512, 512)

# Matching & Attendance
SIMILARITY_THRESHOLD = 0.60
PRESENCE_SECONDS = 60
MIN_IDENTITY_CONFIRMATIONS = 3

# Privacy
STORE_SNAPSHOTS = False
```

---

## File Structure (Planned)

```
Cloned-MVP/
├── app/
│   ├── __init__.py
│   ├── config.py          # NEW: All configuration
│   ├── tracker.py         # NEW: ByteTrack wrapper + TrackState
│   ├── tiling.py          # NEW: Tiled detection
│   └── roi_zoom.py        # NEW: ROI upscaling
├── tools/
│   └── eval_basic.py      # NEW: Metrics evaluation
├── debug/
│   ├── crops/             # NEW: Debug crops
│   └── tile_overview/     # NEW: Tile visualizations
├── attendance.csv         # NEW: Attendance records
├── attendance.sqlite      # NEW: Attendance database
├── main.py                # MODIFIED: Integration
├── pyproject.toml         # MODIFIED: Dependencies
└── README.md              # MODIFIED: Documentation
```

---

## Risk Assessment

### High Risk
- **ByteTrack Integration**: May have dependency conflicts
  - Mitigation: Implement IOU tracker fallback first

### Medium Risk
- **Performance on GTX 1650**: Tiling + ROI may strain GPU
  - Mitigation: Aggressive throttling, batch size limits
- **ID Switch Rate**: May not reach ≤1/min target
  - Mitigation: Tune TRACK_INACTIVE_TTL and EMBED_ROLLING_WINDOW

### Low Risk
- **FAISS Integration**: Already well-implemented
- **ONNX Models**: Already working correctly

---

## Testing Strategy

### Unit Tests
1. Tiling: Verify tile generation for standard resolutions
2. ROI Zoom: Verify coordinate mapping back to frame
3. TrackState: Verify embedding averaging and normalization

### Integration Tests
1. Tracker: Verify stable IDs across frames
2. Tiled Detection: Verify NMS merge correctness
3. Attendance: Verify time-based marking logic

### End-to-End Tests
1. Smoke: Webcam with real-time display
2. Tiling: vid_far.mp4 triggers tiling and improves recall
3. ROI: Small faces get zoomed and improved
4. Attendance: Records created after presence threshold

---

## Open Questions

1. ❓ Should we use motpy or custom IOU tracker as fallback?
   - Decision: Start with motpy (already in dependencies)

2. ❓ What format for attendance.sqlite schema?
   - Decision: TBD in Phase 4

3. ❓ Should SR (super-resolution) be integrated now or later?
   - Decision: Later (disabled for now, keep hooks)

---

## Resources

- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- SCRFD Paper: https://arxiv.org/abs/2105.04714
- ArcFace Paper: https://arxiv.org/abs/1801.07698
- FAISS Documentation: https://faiss.ai/

---

*Last Updated: November 3, 2025*
