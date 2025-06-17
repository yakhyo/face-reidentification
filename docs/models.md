# Model Information

This document provides detailed information about the models used in the Face Re-Identification project.

## Face Detection Models (SCRFD)

The project uses the SCRFD (Sample and Computation Redistribution for Efficient Face Detection) model for face detection. The implementation uses a common SCRFD class that can load any of the model variants below. The model variants differ in size and accuracy but share the same architecture and API.

### SCRFD-500M
- **File**: `det_500m.onnx`
- **Size**: 2.41 MB
- **Characteristics**:
  - Lightweight model
  - Suitable for mobile and edge devices
  - Good balance of speed and accuracy
  - Recommended for real-time applications with limited resources

### SCRFD-2.5G
- **File**: `det_2.5g.onnx`
- **Size**: 3.14 MB
- **Characteristics**:
  - Medium-sized model
  - Better accuracy than 500M variant
  - Suitable for desktop applications
  - Good for general-purpose face detection

### SCRFD-10G
- **File**: `det_10g.onnx`
- **Size**: 16.1 MB
- **Characteristics**:
  - Large model with high accuracy
  - Best for offline processing
  - Suitable when accuracy is critical
  - Can detect smaller and more challenging faces

## Face Recognition Models (ArcFace)

The project uses ArcFace for face recognition and embedding generation. The implementation uses a common ArcFace class that can load any of the model variants below. Both variants produce 512-dimensional face embeddings but differ in their backbone architecture, size, and accuracy.

### ArcFace ResNet-50
- **File**: `w600k_r50.onnx`
- **Size**: 166 MB
- **Characteristics**:
  - High-accuracy model
  - ResNet-50 backbone
  - 512-dimensional face embeddings
  - Recommended for applications requiring highest accuracy

### ArcFace MobileFace
- **File**: `w600k_mbf.onnx`
- **Size**: 12.99 MB
- **Characteristics**:
  - Lightweight model
  - MobileFaceNet backbone
  - 512-dimensional face embeddings
  - Suitable for mobile and edge devices

## Model Integration

All models in the project are integrated with:
- Comprehensive logging system for monitoring and debugging
- ONNX Runtime for efficient inference
- Support for both CPU and GPU execution via ONNX Runtime providers
- Automated model weight management through the download.sh script
- Integration with FAISS for efficient similarity search in the FaceDatabase class

## Performance Considerations

### Estimated Performance

The following performance metrics are estimates based on typical hardware configurations and are provided as a general guideline. Actual performance may vary significantly depending on your specific hardware, input image size, and other factors.

### Face Detection (Estimated)
- SCRFD-500M: ~15-20ms on CPU
- SCRFD-2.5G: ~25-30ms on CPU
- SCRFD-10G: ~40-50ms on CPU

### Face Recognition (Estimated)
- ResNet-50: ~25-30ms on CPU
- MobileFace: ~10-15ms on CPU

*Note: These metrics are estimates only and have not been benchmarked in the current codebase. Your actual performance may vary based on hardware configuration, image size, and other factors.*

## Model Selection Guide

1. **Resource-Constrained Environments** (Mobile/Edge):
   - Detection: SCRFD-500M
   - Recognition: ArcFace MobileFace

2. **Balanced Performance** (Desktop):
   - Detection: SCRFD-2.5G
   - Recognition: ArcFace MobileFace/ResNet-50

3. **High Accuracy Requirements**:
   - Detection: SCRFD-10G
   - Recognition: ArcFace ResNet-50
