import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert YOLOv8 ONNX to RKNN for rk3588.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--output", required=True, help="Output .rknn file.")
    parser.add_argument("--target", default="rk3588", help="Rockchip target platform.")
    parser.add_argument(
        "--quantized-dataset",
        default="",
        help="Path to image list txt for INT8 quantization. Leave empty for non-quantized build.",
    )
    parser.add_argument("--mean-values", default="0 0 0", help="Space-separated RGB mean.")
    parser.add_argument("--std-values", default="255 255 255", help="Space-separated RGB std.")
    return parser


def parse_triplet(raw_value: str) -> list[list[float]]:
    parts = [float(item) for item in raw_value.split()]
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 numeric values.")
    return [parts]


def main() -> None:
    args = build_arg_parser().parse_args()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError("rknn-toolkit2 is required on the conversion machine.") from exc

    quantized = bool(args.quantized_dataset)
    dataset_path = Path(args.quantized_dataset).resolve() if quantized else None
    if dataset_path and not dataset_path.exists():
        raise FileNotFoundError(f"Quantization dataset txt not found: {dataset_path}")

    rknn = RKNN(verbose=True)
    mean_values = parse_triplet(args.mean_values)
    std_values = parse_triplet(args.std_values)

    print("Configuring RKNN...")
    ret = rknn.config(
        target_platform=args.target,
        mean_values=mean_values,
        std_values=std_values,
        quantized_algorithm="normal",
        optimization_level=3,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    print("Loading ONNX...")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    print(f"Building RKNN (quantized={quantized})...")
    ret = rknn.build(
        do_quantization=quantized,
        dataset=str(dataset_path) if dataset_path else None,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code {ret}")

    output_path = Path(args.output).resolve()
    print(f"Exporting RKNN to {output_path} ...")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")

    rknn.release()
    print(output_path)


if __name__ == "__main__":
    main()
