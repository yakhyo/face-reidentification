#!/bin/bash

# Check if the weights directory does not exist, then create it
if [ ! -d "weights" ]; then
  mkdir weights
fi

# Clean the weights directory
rm -rf weights/*.onnx

# Download the file and save it to the weights directory
wget -O weights/w600k_r50.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx
wget -O weights/det_10g.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx