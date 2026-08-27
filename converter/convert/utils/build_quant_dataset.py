import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build image list for RKNN quantization.")
    parser.add_argument("--images", required=True, help="Directory with calibration images.")
    parser.add_argument("--output", default="calibration_images.txt", help="Output txt file.")
    parser.add_argument("--limit", type=int, default=300, help="Max number of images.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    images_dir = Path(args.images).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = [
        p for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths = image_paths[: max(1, args.limit)]

    output_path = Path(args.output).resolve()
    output_path.write_text(
        "\n".join(str(path) for path in image_paths) + ("\n" if image_paths else ""),
        encoding="utf-8",
    )
    print(f"Saved {len(image_paths)} entries to {output_path}")


if __name__ == "__main__":
    main()
