
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR =  PROJECT_DIR / "model"
DATASET_DIR = MODEL_DIR / "dataset"
TARGET_DIR = MODEL_DIR / "target"

BEST_PT_DIR = TARGET_DIR / "best.pt"
BEST_ONNX_DIR = TARGET_DIR / "best.onnx"
BEST_RKNN_DIR = TARGET_DIR / "best.rknn"

IMAGES_DIR = "images" 
LABELS_DIR = "labels"

TRAIN_DIR = "train"
VAL_DIR = "val"

CLASSES_DIR = "classes.txt"