import argparse
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from src.data.loader import read_split_file, read_labels_csv
from src.deep.cnn_embedder import build_efficientnet_b0_embedder
from PIL import Image


@torch.inference_mode()
def embed_split(dataset_root: Path, split: str, batch_size: int, device: torch.device, out_dir: Path) -> None:
    console = Console()
    model, preprocess, emb_dim = build_efficientnet_b0_embedder(device)

    paths = read_split_file(dataset_root, split)
    labels_map = read_labels_csv(dataset_root)
    images, labels = [], []
    rel_paths: list[str] = []
    embs_list = []

    def flush_batch() -> None:
        nonlocal images, labels
        if not images:
            return
        batch = torch.stack(images, dim=0).to(device)
        feats = model(batch)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        embs_list.append(feats.cpu().numpy())
        images.clear()
        labels.clear()

    count = 0
    y_all = []
    for abs_path in paths:
        rel = abs_path.relative_to(dataset_root).as_posix()
        label = labels_map.get(rel)
        if label is None:
            continue
        # Load image
        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            t = preprocess(im)
            images.append(t)
            y_all.append(label)
            rel_paths.append(rel)
        count += 1
        if len(images) >= batch_size:
            flush_batch()
            console.print(f"{split}: processed {count}")

    flush_batch()
    X = np.concatenate(embs_list, axis=0).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    assert X.shape[1] == emb_dim, f"Unexpected embedding dim: {X.shape}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_X.npy", X)
    np.save(out_dir / f"{split}_y.npy", y)
    # Save corresponding relative paths for row->image mapping
    (out_dir / f"{split}_paths.txt").write_text("\n".join(rel_paths))
    console.print(f"Saved {split}: X.shape={X.shape}, y.shape={y.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/teamspace/studios/this_studio/dataset")
    parser.add_argument("--out_root", type=str, default="/teamspace/studios/this_studio/dataset/features_cnn")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"], choices=["train", "val", "test"]) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    for split in args.splits:
        embed_split(root, split, args.batch_size, device, out_root)


if __name__ == "__main__":
    main()


