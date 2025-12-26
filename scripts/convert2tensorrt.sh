#!/usr/bin/bash

ONNX_PATH=$1
OUTPUT_DIR=$2
RESULT_MODEL_NAME=$3

echo "Converting $ONNX_PATH to TensorRT ($PRECISION)..."

mkdir -p $OUTPUT_DIR
/usr/src/tensorrt/bin/trtexec --onnx=$ONNX_PATH --saveEngine=$OUTPUT_DIR/${RESULT_MODEL_NAME}.trt --inputIOFormats=fp32:chw
