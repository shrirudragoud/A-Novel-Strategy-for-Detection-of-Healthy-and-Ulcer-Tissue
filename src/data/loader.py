from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import csv
import numpy as np
from PIL import Image


def read_split_file(dataset_root: Path, split_name: str) -> List[Path]:
    split_path = dataset_root / "splits" / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    rel_paths = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    return [dataset_root / p for p in rel_paths]


def read_labels_csv(dataset_root: Path) -> Dict[str, int]:
    labels_path = dataset_root / "metadata" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    mapping: Dict[str, int] = {}
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filepath"]] = int(row["label"])
    return mapping


def load_image_rgb(path: Path, resize_hw: Tuple[int, int] = (128, 128)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if resize_hw:
            im = im.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


def iter_split_samples(
    dataset_root: Path,
    split_name: str,
    resize_hw: Tuple[int, int] = (128, 128),
) -> Iterable[Tuple[np.ndarray, int, str]]:
    """
    Yields tuples of (rgb_image_uint8, label_id, relative_path_str)
    """
    paths = read_split_file(dataset_root, split_name)
    labels_map = read_labels_csv(dataset_root)
    for abs_path in paths:
        rel = abs_path.relative_to(dataset_root).as_posix()
        label = labels_map.get(rel)
        if label is None:
            # Skip if label missing
            continue
        rgb = load_image_rgb(abs_path, resize_hw=resize_hw)
        yield rgb, label, rel


