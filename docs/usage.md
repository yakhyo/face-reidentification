# Usage Guide

This guide explains how to use the Face Re-Identification system effectively.

## Basic Usage

The system supports three main operation modes:
- Adding faces to the database
- Real-time face recognition from camera
- Processing video files

### Database Operations

1. **Add Faces to Database**
   ```bash
   # Add a single face
   python main.py --mode add --image assets/faces/person1.jpg --name "Person 1"

   # Add multiple faces from a directory
   python main.py --mode add --dir assets/faces/ --auto-name
   ```

2. **List Database Contents**
   ```bash
   python main.py --mode list
   ```

### Face Recognition

1. **Using Webcam**
   ```bash
   python main.py --mode camera --source 0
   ```

2. **Process Video File**
   ```bash
   python main.py --mode video --input assets/video.mp4 --output results.mp4
   ```

## Command Line Arguments

```bash
usage: main.py [-h] [--mode {add,list,camera,video}]
               [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT]
               [--similarity-thresh SIMILARITY_THRESH]
               [--confidence-thresh CONFIDENCE_THRESH]
               [--input INPUT] [--output OUTPUT]
               [--name NAME] [--auto-name]
               [--max-faces MAX_FACES] [--log-level LOG_LEVEL]
               [--log-file LOG_FILE]
```

### Required Arguments
- `--mode`: Operation mode (add/list/camera/video)
- `--input`: Input source for video mode or image file for add mode

### Optional Arguments
- `--det-weight`: Path to detection model (default: `weights/det_10g.onnx`)
- `--rec-weight`: Path to recognition model (default: `weights/w600k_r50.onnx`)
- `--similarity-thresh`: Face similarity threshold (default: 0.4)
- `--confidence-thresh`: Detection confidence threshold (default: 0.5)
- `--output`: Output video file path (for video mode)
- `--name`: Person's name when adding to database
- `--auto-name`: Use filenames as person names when adding directory
- `--max-faces`: Maximum faces to detect per frame (default: 0, no limit)
- `--log-level`: Logging level (default: "INFO")
- `--log-file`: Log file path for file logging

## Examples

1. **Database Management**
   ```bash
   # Add faces from a directory using filenames as names
   python main.py --mode add --dir assets/faces/ --auto-name

   # Add a single face with custom name
   python main.py --mode add --image assets/faces/john.jpg --name "John Doe"

   # List all faces in database
   python main.py --mode list
   ```

2. **Real-time Recognition**
   ```bash
   # Using webcam with default settings
   python main.py --mode camera --source 0

   # Using webcam with custom thresholds
   python main.py --mode camera --source 0 --similarity-thresh 0.6 --confidence-thresh 0.7
   ```

3. **Video Processing**
   ```bash
   # Process video file and save output
   python main.py --mode video --input assets/video.mp4 --output results.mp4

   # Process video with logging to file
   python main.py --mode video --input video.mp4 --log-level DEBUG --log-file processing.log
   ```

## Logging Configuration

The system supports comprehensive logging for debugging and monitoring:

```bash
# Console-only logging with DEBUG level
python main.py --mode video --input video.mp4 --log-level DEBUG

# Log to both console and file
python main.py --mode video --input video.mp4 --log-level INFO --log-file face_reid.log
```
