import argparse
from pathlib import Path

import numpy as np

from src.data.loader import iter_split_samples
from src.preprocess.image_ops import preprocess_luminance
from src.features.emd_features import extract_emd_features


def process_split(dataset_root: Path, split: str, height: int, width: int, colorspace: str, kernel: int, max_imfs: int, projections: list[str]) -> None:
    X: list[np.ndarray] = []
    y: list[int] = []
    count = 0
    for rgb, label, rel in iter_split_samples(dataset_root, split, resize_hw=(height, width)):
        lum = preprocess_luminance(rgb, colorspace=colorspace, kernel_size=kernel)
        feats = extract_emd_features(lum, max_imfs=max_imfs, projections=projections)
        X.append(feats)
        y.append(label)
        count += 1
        if count % 1000 == 0:
            print(f"{split}: processed {count} images")
    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    out_dir = dataset_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_X.npy", X_arr)
    np.save(out_dir / f"{split}_y.npy", y_arr)
    print(f"Saved {split} features: X.shape={X_arr.shape}, y.shape={y_arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/teamspace/studios/this_studio/dataset")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"], choices=["train", "val", "test"]) 
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--colorspace", type=str, default="HSV", choices=["HSV", "YCbCr"])
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--max_imfs", type=int, default=4)
    parser.add_argument("--projections", type=str, default="rows_mean,cols_mean", help="comma-separated: rows_mean,cols_mean,rows_median,cols_median")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    projections = [s.strip() for s in args.projections.split(",") if s.strip()]
    for split in args.splits:
        process_split(root, split, args.height, args.width, args.colorspace, args.kernel, args.max_imfs, projections)


if __name__ == "__main__":
    main()


