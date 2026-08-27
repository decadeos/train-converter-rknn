import sys
import subprocess

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))            
import constant.general as CG
import constant.model as CM


def run_command(args, description):
    """Helper to run shell commands and catch errors."""
    print(f"[ForOrange] {description}...")
    try:
        # Use shell=True on Windows for built-in commands, False for executables
        subprocess.run(args, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        sys.exit(1)



def main():

    print(f"[ForOrange] Python: {CG.PYTHON}")

    script_dir = Path(__file__).parent.absolute()
    utils_dir = script_dir / "utils"

    # 3. Export PT to ONNX
    export_script = utils_dir / "export_best_to_onnx.py"
    
    run_command(
        [str(CG.PYTHON), str(export_script), "--weights", str(CM.BEST_PT_DIR), "--imgsz", "320"],
        "Exporting best.pt -> ONNX"
    )

    # 4. Verify ONNX File Existence
    onnx_path = CM.BEST_ONNX_DIR
    if not onnx_path.exists():
        print(f"ONNX export failed: {onnx_path} not found")
        sys.exit(1)

    # 5. Graph Verification
    verify_cmd = f"import onnx; m=onnx.load(r'{onnx_path}'); onnx.checker.check_model(m); print('ONNX_OK', [i.name for i in m.graph.input])"
    run_command([str(CG.PYTHON), "-c", verify_cmd], "Verifying ONNX model graph")

    # 6. ONNXRuntime Smoke Test
    sample_image = CM.DATASET_DIR / "images" / "object_262.jpg"
    if sample_image.exists():
        smoke_test_code = f"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

model_path = r"{onnx_path}"
image_path = r"{sample_image}"
image = cv2.imread(image_path)
if image is None: raise SystemExit("Could not read image")

size = 320
h, w = image.shape[:2]
scale = min(size / w, size / h)
nw, nh = int(round(w * scale)), int(round(h * scale))
resized = cv2.resize(image, (nw, nh))
canvas = np.full((size, size, 3), 114, dtype=np.uint8)
canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = resized
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = np.transpose(rgb, (2, 0, 1))[None, ...]

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
outputs = sess.run(None, {{sess.get_inputs()[0].name: tensor}})
print("ONNXRUNTIME_OK", "Max value:", float(np.max(outputs[0])))
"""
        run_command([str(CG.PYTHON), "-c", smoke_test_code], "Running ONNXRuntime smoke test")
    else:
        print("[ForOrange] Sample image not found, skipping smoke test")

    # # 7. Build Quantization List
    # quant_list = script_dir / "calibration_images.txt"
    # img_dir = CM.DATASET_DIR / CM.IMAGES_DIR
    # build_quant_script = utils_dir / "build_quant_dataset.py"
    
    # run_command(
    #     [str(CG.PYTHON), str(build_quant_script), "--images", str(img_dir), "--output", str(quant_list), "--limit", "300"],
    #     "Building quantization dataset list"
    # )

    # 8. RKNN Conversion (Check if library exists)
    check_rknn = "import importlib.util; print('1' if importlib.util.find_spec('rknn') else '0')"
    has_rknn = subprocess.check_output([str(CG.PYTHON), "-c", check_rknn]).decode().strip()

    if has_rknn == "1":
        build_rknn_script = utils_dir / "build_rknn_model.py"
        run_command(
            [str(CG.PYTHON), str(build_rknn_script), 
             "--onnx", str(onnx_path), 
             "--output", str(CM.BEST_RKNN_DIR), 
             "--target", "rk3588", 
            #  "--quantized-dataset", str(quant_list)
            ],
            "Building RKNN model"
        )
    else:
        print("[ForOrange] rknn-toolkit2 not found, skipping RKNN build")

if __name__ == "__main__":
    main()