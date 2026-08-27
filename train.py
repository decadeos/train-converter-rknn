import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_device(preferred: str) -> str:
    cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if preferred == "auto":
        return "0" if cuda_ready else "cpu"
    if preferred in {"0", "cuda"} and not cuda_ready:
        raise RuntimeError(
            "GPU was requested, but CUDA is not available. Check the NVIDIA driver "
            "and the torch CUDA installation."
        )
    return preferred


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, list):
        return {class_id: str(name) for class_id, name in enumerate(names)}

    raise ValueError(f"{data_yaml} must contain names as a list or a dict.")


def collect_label_class_ids(labels_dir: Path) -> set[int]:
    class_ids: set[int] = set()
    bad_lines: list[str] = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.name == "classes.txt":
            continue

        with label_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                try:
                    class_ids.add(int(parts[0]))
                except (ValueError, IndexError):
                    bad_lines.append(f"{label_file}:{line_no}: {line}")

    if bad_lines:
        examples = "\n".join(bad_lines[:10])
        raise ValueError(f"Invalid YOLO label lines:\n{examples}")

    return class_ids


def validate_dataset(data_yaml: Path) -> None:
    names = load_class_names(data_yaml)
    label_class_ids = collect_label_class_ids(Path("dataset/labels"))
    unknown_ids = sorted(label_class_ids - set(names))

    if unknown_ids:
        known = ", ".join(f"{class_id}: {name}" for class_id, name in sorted(names.items()))
        raise ValueError(
            "Labels contain class ids missing from data.yaml: "
            f"{unknown_ids}. Current data.yaml classes: {known}"
        )

    print("data.yaml classes:")
    for class_id, name in sorted(names.items()):
        suffix = " (present in labels)" if class_id in label_class_ids else " (no objects)"
        print(f"  {class_id}: {name}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", default="auto", help="auto | 0 | cpu")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    validate_dataset(data_yaml)

    device = resolve_device(args.device)
    workers = 4 if device != "cpu" else 0
    batch = 8 if device != "cpu" else 4

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        single_cls=True,
        epochs=3,
        imgsz=640,
        batch=batch,
        device=device,
        workers=workers,
        project=os.path.join(SCRIPT_DIR, "model"),
        name="target",
        exist_ok=True,
        verbose=False,
        plots=False,
        save=False,
    )


if __name__ == "__main__":
    main()
