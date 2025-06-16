# Model Information

This document provides detailed information about the models used in the Face Re-Identification project.

## Face Detection Models (SCRFD)

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

### ArcFace ResNet-50
- **File**: `w600k_r50.onnx`
- **Size**: 166 MB
- **Characteristics**:
  - High-accuracy model
  - ResNet-50 backbone
  - 512-dimensional face embeddings
  - Recommended for applications requiring highest accuracy
- **Features**:
  - Integrated logging for better monitoring
  - Improved error handling and validation
  - Optimized face alignment process
  - Support for FAISS-based similarity search

### ArcFace MobileFace
- **File**: `w600k_mbf.onnx`
- **Size**: 12.99 MB
- **Characteristics**:
  - Lightweight model
  - MobileFaceNet backbone
  - 512-dimensional face embeddings
  - Suitable for mobile and edge devices
- **Features**:
  - Efficient inference on CPU
  - Optimized for real-time processing
  - Good balance of speed and accuracy

## Model Integration

All models are integrated with:
- Comprehensive logging system for monitoring and debugging
- ONNX Runtime for efficient inference
- Support for both CPU and GPU execution
- Automated model weight management
- Integration with FAISS for efficient similarity search

## Performance Considerations

### Face Detection
- SCRFD-500M: ~15-20ms on CPU
- SCRFD-2.5G: ~25-30ms on CPU
- SCRFD-10G: ~40-50ms on CPU

### Face Recognition
- ResNet-50: ~25-30ms on CPU
- MobileFace: ~10-15ms on CPU

*Note: Performance metrics are approximate and may vary based on hardware configuration.*

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
