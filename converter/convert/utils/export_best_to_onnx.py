import argparse
from pathlib import Path

from ultralytics import YOLO


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ../best.pt to ONNX for RKNN conversion.")
    parser.add_argument("--weights", default="../best.pt", help="Path to .pt weights.")
    parser.add_argument("--imgsz", type=int, default=320, help="Square inference size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument("--device", default="cpu", help="Export device.")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX when supported.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
        half=args.half,
        device=args.device,
    )
    print(Path(export_path).resolve())


if __name__ == "__main__":
    main()
