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
  - Better accuracy than 500M
  - Still maintains good inference speed
  - Recommended for general-purpose applications

### SCRFD-10G
- **File**: `det_10g.onnx`
- **Size**: 16.1 MB
- **Characteristics**:
  - High-accuracy model
  - More computational demands
  - Best detection performance
  - Recommended for accuracy-critical applications

## Face Recognition Models (ArcFace)

### ArcFace MobileFace
- **File**: `w600k_mbf.onnx`
- **Size**: 12.99 MB
- **Characteristics**:
  - Lightweight backbone
  - Efficient computation
  - Suitable for mobile devices
  - Good accuracy for most applications

### ArcFace ResNet-50
- **File**: `w600k_r50.onnx`
- **Size**: 166 MB
- **Characteristics**:
  - ResNet-50 backbone
  - High accuracy
  - More computational demands
  - State-of-the-art recognition performance

## Model Selection Guide

### Use Case Recommendations

1. **Resource-Constrained Environments**
   - Detection: SCRFD-500M
   - Recognition: ArcFace MobileFace
   - Total Size: ~15.4 MB

2. **Balanced Performance**
   - Detection: SCRFD-2.5G
   - Recognition: ArcFace MobileFace
   - Total Size: ~16.13 MB

3. **High Accuracy Requirements**
   - Detection: SCRFD-10G
   - Recognition: ArcFace ResNet-50
   - Total Size: ~182.1 MB

### Performance Characteristics

1. **Speed Priority**
   ```
   SCRFD-500M + MobileFace
   - Fastest inference
   - Lowest resource usage
   - Suitable for real-time applications
   ```

2. **Balanced**
   ```
   SCRFD-2.5G + MobileFace
   - Good accuracy
   - Reasonable speed
   - Moderate resource usage
   ```

3. **Accuracy Priority**
   ```
   SCRFD-10G + ResNet-50
   - Highest accuracy
   - Higher resource requirements
   - Suitable for offline processing
   ```

## Model Training Information

The models are trained on large-scale face datasets:

- Detection models: Trained on WIDER FACE dataset
- Recognition models: Trained on MS1M-v3 dataset

## Model References

1. SCRFD Paper:
   - Title: "Sample and Computation Redistribution for Efficient Face Detection"
   - URL: https://arxiv.org/abs/2105.04714

2. ArcFace Paper:
   - Title: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
   - URL: https://arxiv.org/abs/1801.07698
