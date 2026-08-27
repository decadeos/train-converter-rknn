#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv-linux}"
IMG_SIZE="${IMG_SIZE:-320}"
QUANT_LIMIT="${QUANT_LIMIT:-300}"

echo "[ForOrange] Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  rm -rf "$VENV_DIR"
  echo "[ForOrange] Creating virtualenv at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[ForOrange] Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "[ForOrange] Pinning setuptools for RKNN Toolkit2 compatibility"
python -m pip install "setuptools<81"

ONNX_PATH="$PROJECT_ROOT/model/target/best.onnx"
if [ ! -f "$ONNX_PATH" ]; then
  echo "[ForOrange] ERROR: best.onnx not found: $ONNX_PATH" >&2
  echo "[ForOrange] First create ONNX on Windows with convert_on_windows.ps1" >&2
  exit 1
fi

echo "[ForOrange] Installing Linux conversion dependencies"
python -m pip install \
  "numpy<=1.26.4" \
  "protobuf>=4.21.6,<=4.25.4" \
  "onnx==1.16.1" \
  "onnxruntime>=1.17.0" \
  "opencv-python==4.11.0.86" \
  "ruamel.yaml>=0.17.21" \
  "tqdm>=4.64.1" \
  "fast-histogram>=0.11"

echo "[ForOrange] Installing RKNN Toolkit2"
python -m pip install --no-deps rknn-toolkit2

echo "[ForOrange] Verifying ONNX"
python - <<PY
import onnx
from pathlib import Path
path = Path(r"$ONNX_PATH")
m = onnx.load(path)
onnx.checker.check_model(m)
print("ONNX_OK", [i.name for i in m.graph.input], [o.name for o in m.graph.output])
PY

SAMPLE_IMAGE="$PROJECT_ROOT/model/dataset/images/object_262.jpg"
if [ -f "$SAMPLE_IMAGE" ]; then
  echo "[ForOrange] Running ONNXRuntime smoke test"
  python - <<PY
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

model_path = Path(r"$ONNX_PATH")
image_path = Path(r"$SAMPLE_IMAGE")
image = cv2.imread(str(image_path))
size = int(r"$IMG_SIZE")
h, w = image.shape[:2]
scale = min(size / w, size / h)
nw, nh = int(round(w * scale)), int(round(h * scale))
resized = cv2.resize(image, (nw, nh))
canvas = np.full((size, size, 3), 114, dtype=np.uint8)
px = (size - nw) // 2
py = (size - nh) // 2
canvas[py:py + nh, px:px + nw] = resized
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
outputs = sess.run(None, {sess.get_inputs()[0].name: tensor})
print("ONNXRUNTIME_OK", image.shape, [o.shape for o in outputs], float(np.max(outputs[0])))
PY
fi

CALIB_PATH="$SCRIPT_DIR/calibration_images_linux.txt"
echo "[ForOrange] Building calibration list"
python "$SCRIPT_DIR/utils/build_quant_dataset.py" --images "$PROJECT_ROOT/model/dataset/images" --output "$CALIB_PATH" --limit "$QUANT_LIMIT"

RKNN_PATH="$SCRIPT_DIR/best_${IMG_SIZE}_i8.rknn"
echo "[ForOrange] Building RKNN"
python "$SCRIPT_DIR/utils/build_rknn_model.py" \
  --onnx "$ONNX_PATH" \
  --output "$RKNN_PATH" \
  --target rk3588 \
  --quantized-dataset "$CALIB_PATH"

echo "[ForOrange] Done"
echo "[ForOrange] ONNX: $ONNX_PATH"
echo "[ForOrange] RKNN: $RKNN_PATH"
