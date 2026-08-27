#!/usr/bin/env bash
set -euo pipefail

# Recreate project-local virtual environment.
python3 -m venv ./venv
PY="./venv/bin/python"
PIP="$PY -m pip"

$PIP install --upgrade pip wheel
$PIP install setuptools==80.9.0

# Base tooling used by label/utility scripts.
$PIP install PyQt5==5.15.11 lxml==6.1.1 paramiko==4.0.0 opencv-python==4.11.0.86

# RKNN toolkit 2.3.2 has strict upper bounds. Keep these pinned before YOLO.
$PIP install \
  numpy==1.26.4 \
  protobuf==4.25.4 \
  onnx==1.16.1 \
  onnxruntime==1.25.0 \
  onnxslim==0.1.91

# rknn-toolkit2 requires torch <= 2.4.0. Do not install PyTorch nightly here.
$PIP install torch==2.4.0+cpu torchvision==0.19.0+cpu --index-url https://download.pytorch.org/whl/cpu

$PIP install rknn-toolkit2==2.3.2

# Runtime deps for Ultralytics, pinned explicitly to avoid resolver drift.
$PIP install \
  matplotlib==3.10.9 \
  polars==1.40.1 \
  pyyaml==6.0.3 \
  requests==2.33.1

# Install Ultralytics after the pinned RKNN stack, without letting it upgrade deps.
$PIP install ultralytics==8.4.41 ultralytics-thop==2.0.19 --no-deps

$PY -m pip check
