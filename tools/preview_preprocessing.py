import argparse
from pathlib import Path

import numpy as np

from src.data.loader import iter_split_samples
from src.preprocess.image_ops import preprocess_luminance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/teamspace/studios/this_studio/dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--colorspace", type=str, default="HSV", choices=["HSV", "YCbCr"])
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    hs, ws = args.height, args.width
    count = 0
    per_class = {}
    for rgb, label, rel in iter_split_samples(root, args.split, resize_hw=(hs, ws)):
        lum = preprocess_luminance(rgb, colorspace=args.colorspace, kernel_size=args.kernel)
        assert lum.shape == (hs, ws), f"Unexpected luminance shape: {lum.shape}"
        per_class[label] = per_class.get(label, 0) + 1
        count += 1
        if count <= 3:
            print(f"Sample {count}: {rel}, label={label}, lum_range=({float(lum.min()):.3f},{float(lum.max()):.3f})")
        if count >= args.max_samples:
            break
    print(f"Processed {count} samples from split={args.split}. Class counts: {per_class}")


if __name__ == "__main__":
    main()


