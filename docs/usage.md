# Usage Guide

This guide explains how to use the Face Re-Identification system effectively.

## Basic Usage

1. **Prepare Target Faces**
   - Place face images in the `faces` directory
   - Use clear, front-facing photos
   - Name files according to the person's identity (e.g., `john.jpg`, `jane.png`)

2. **Run the System**
   ```bash
   # Basic usage with default parameters
   python main.py --source <video_source>
   ```

## Command Line Arguments

```bash
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT]
               [--similarity-thresh SIMILARITY_THRESH]
               [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE]
               [--max-num MAX_NUM] [--log-level LOG_LEVEL]
```

### Required Arguments
- `--source`: Input source (video file path or camera index)

### Optional Arguments
- `--det-weight`: Path to detection model (default: `weights/det_10g.onnx`)
- `--rec-weight`: Path to recognition model (default: `weights/w600k_r50.onnx`)
- `--similarity-thresh`: Face similarity threshold (default: 0.4)
- `--confidence-thresh`: Detection confidence threshold (default: 0.5)
- `--faces-dir`: Directory containing target faces (default: `faces`)
- `--max-num`: Maximum faces to detect per frame (default: 0, no limit)
- `--log-level`: Logging level (default: "INFO")

## Examples

1. **Using Webcam**
   ```bash
   python main.py --source 0
   ```

2. **Process Video File**
   ```bash
   python main.py --source assets/video.mp4
   ```

3. **Adjust Detection Sensitivity**
   ```bash
   python main.py --source 0 --confidence-thresh 0.7
   ```

4. **Use Lightweight Models**
   ```bash
   python main.py --source 0 --det-weight weights/det_500m.onnx --rec-weight weights/w600k_mbf.onnx
   ```

## Output Interpretation

- Green bounding box: Known face detected
- Blue bounding box: Unknown face detected
- Text display: `Name: Similarity_Score`
- Vertical bar: Visual representation of similarity score

## Performance Tips

1. **Model Selection**
   - Use smaller models (500M/MobileFace) for speed
   - Use larger models (10G/ResNet-50) for accuracy

2. **Threshold Adjustment**
   - Increase `confidence-thresh` to reduce false detections
   - Adjust `similarity-thresh` to control recognition sensitivity

3. **Resource Usage**
   - Limit `max-num` on lower-end systems
   - Consider input resolution for performance

## Keyboard Controls

- Press 'q' to quit the application
- The output video is automatically saved as 'output_video.mp4'
