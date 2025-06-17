# Usage Guide

This guide explains how to use the Face Re-Identification system effectively.

## Basic Usage

The system supports real-time face recognition and database management operations.

### Database Operations

1. **Add Faces to Database**
   ```bash
   # Add faces from a directory
   python main.py --faces-dir assets/faces --update-db
   ```

2. **Using Pre-built Database**
   ```bash
   # Use existing database without updating
   python main.py --db-path ./database/face_database
   ```

### Face Recognition

1. **Using Webcam**
   ```bash
   python main.py --source 0
   ```

2. **Process Video File**
   ```bash
   python main.py --source assets/in_video.mp4 --output result.mp4
   ```

## Command Line Arguments

```bash
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT]
               [--similarity-thresh SIMILARITY_THRESH]
               [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE]
               [--max-num MAX_NUM] [--db-path DB_PATH]
               [--update-db] [--output OUTPUT]
```

### Required Arguments
None (all arguments have default values)

### Optional Arguments
- `--det-weight`: Path to detection model (default: `./weights/det_10g.onnx`)
- `--rec-weight`: Path to recognition model (default: `./weights/w600k_mbf.onnx`)
- `--similarity-thresh`: Face similarity threshold (default: 0.4)
- `--confidence-thresh`: Detection confidence threshold (default: 0.5)
- `--faces-dir`: Directory containing face images (default: `./assets/faces`)
- `--source`: Input source - video file or camera index (default: `./assets/in_video.mp4`)
- `--max-num`: Maximum faces to detect per frame (default: 0, no limit)
- `--db-path`: Path to face database (default: `./database/face_database`)
- `--update-db`: Force creation or update of the face database. Required when building the database for the first time. If not specified and a database already exists, the system will load the existing database instead of rebuilding it. (flag)
- `--output`: Output video path (default: `output_video.mp4`)

## Examples

1. **Database Management**
   ```bash
   # Update face database with new images
   python main.py --faces-dir ./assets/faces --update-db

   # Use custom database path
   python main.py --db-path ./custom/database/path
   ```

2. **Real-time Recognition**
   ```bash
   # Using webcam with default settings
   python main.py --source 0

   # Using webcam with custom thresholds
   python main.py --source 0 --similarity-thresh 0.6 --confidence-thresh 0.7
   ```

3. **Video Processing**
   ```bash
   # Process video with custom models
   python main.py --source video.mp4 --det-weight weights/det_500m.onnx --rec-weight weights/w600k_mbf.onnx

   # Limit maximum faces per frame
   python main.py --source video.mp4 --max-num 5
   ```
