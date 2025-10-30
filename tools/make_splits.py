import json
import csv
import random
from pathlib import Path
from typing import Dict, List


def gather_class_files(images_root: Path, class_names: List[str]) -> Dict[str, List[Path]]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    class_to_files: Dict[str, List[Path]] = {}
    for class_name in class_names:
        class_dir = images_root / class_name
        if not class_dir.exists():
            class_to_files[class_name] = []
            continue
        files = [p for p in class_dir.rglob("*") if p.suffix.lower() in valid_exts]
        files.sort()
        class_to_files[class_name] = files
    return class_to_files


def write_class_map(meta_root: Path, class_to_id: Dict[str, int]) -> None:
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "class_map.json").write_text(json.dumps(class_to_id, indent=2))


def write_labels_csv(dataset_root: Path, meta_root: Path, class_to_id: Dict[str, int], class_to_files: Dict[str, List[Path]]) -> None:
    meta_root.mkdir(parents=True, exist_ok=True)
    labels_path = meta_root / "labels.csv"
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])  # relative to dataset root
        for class_name, files in class_to_files.items():
            for p in files:
                rel = p.relative_to(dataset_root).as_posix()
                writer.writerow([rel, class_to_id[class_name]])


def make_stratified_splits(
    class_to_files: Dict[str, List[Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[Path]]:
    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for class_name, files in class_to_files.items():
        files = files.copy()
        random.shuffle(files)
        n = len(files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train = files[:n_train]
        val = files[n_train : n_train + n_val]
        test = files[n_train + n_val :]
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
    return splits


def write_split_files(dataset_root: Path, splits_root: Path, splits: Dict[str, List[Path]]) -> None:
    splits_root.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        rels = [p.relative_to(dataset_root).as_posix() for p in items]
        (splits_root / f"{name}.txt").write_text("\n".join(sorted(rels)))


def print_stats(splits: Dict[str, List[Path]], dataset_root: Path) -> None:
    counts = {k: len(v) for k, v in splits.items()}
    # Derive per-class counts from relative paths: dataset/images/<class>/<file>
    per_class: Dict[str, Dict[str, int]] = {}
    for split_name, items in splits.items():
        for p in items:
            try:
                rel = p.relative_to(dataset_root).as_posix()
                parts = rel.split("/")
                cls = parts[2] if len(parts) > 2 else "unknown"
            except Exception:
                cls = "unknown"
            per_class.setdefault(cls, {"train": 0, "val": 0, "test": 0})
            per_class[cls][split_name] += 1
    print("Split counts:", counts)
    print("Per-class:", per_class)


def main() -> None:
    # Configure here if your layout differs
    dataset_root = Path("/teamspace/studios/this_studio/dataset")
    images_root = dataset_root / "images"
    splits_root = dataset_root / "splits"
    meta_root = dataset_root / "metadata"

    # Adjust if your class names differ
    class_names = ["healthy", "ulcer", "polyp"]
    class_to_id = {c: i for i, c in enumerate(class_names)}

    random.seed(42)

    class_to_files = gather_class_files(images_root, class_names)

    write_class_map(meta_root, class_to_id)
    write_labels_csv(dataset_root, meta_root, class_to_id, class_to_files)

    splits = make_stratified_splits(class_to_files, train_ratio=0.8, val_ratio=0.1)
    write_split_files(dataset_root, splits_root, splits)
    print_stats(splits, dataset_root)

    print("Done.")


if __name__ == "__main__":
    main()


