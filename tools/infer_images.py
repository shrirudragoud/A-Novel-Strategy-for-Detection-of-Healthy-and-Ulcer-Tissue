import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import torch
from PIL import Image

from src.deep.cnn_embedder import build_efficientnet_b0_embedder


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="/teamspace/studios/this_studio/artifacts/cnn_fast_cw")
    parser.add_argument("--images", type=str, nargs="+", help="one or more image paths")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--class_map", type=str, default="/teamspace/studios/this_studio/dataset/metadata/class_map.json")
    parser.add_argument("--out_csv", type=str, default="/teamspace/studios/this_studio/artifacts/infer_results.csv")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load pipeline artifacts
    art = Path(args.artifacts_dir)
    scaler = joblib.load(art / "scaler.pkl")
    pls = joblib.load(art / "pls.pkl")
    svm = joblib.load(art / "svm.pkl")

    # Load class mapping
    with open(args.class_map) as f:
        cls_map = json.load(f)
    id_to_cls = {v: k for k, v in cls_map.items()}

    # Build embedder
    model, preprocess, emb_dim = build_efficientnet_b0_embedder(device)

    # Inference
    rows = []
    for img_path in args.images:
        p = Path(img_path)
        with Image.open(p) as im:
            im = im.convert("RGB")
            t = preprocess(im).unsqueeze(0).to(device)
            feats = model(t)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            x = feats.cpu().numpy()
        x2 = scaler.transform(x)
        x3 = pls.transform(x2)
        pred = int(svm.predict(x3)[0])
        try:
            proba = svm.predict_proba(x3)[0].tolist()
        except Exception:
            proba = []
        rows.append([str(p), id_to_cls.get(pred, str(pred))] + proba)

    # Save CSV
    import csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["path", "predicted"] + [f"proba_{i}" for i in range(len(proba))]
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()


