





import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageEnhance
import torch
import joblib

from src.deep.cnn_embedder import build_efficientnet_b0_embedder


@torch.inference_mode()
def predict_proba(model, preprocess, device, scaler, pls, svm, img: Image.Image) -> np.ndarray:
    t = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    feats = model(t)
    if feats.ndim > 2:
        feats = torch.flatten(feats, 1)
    x = feats.cpu().numpy()
    x2 = scaler.transform(x)
    x3 = pls.transform(x2)
    proba = svm.predict_proba(x3)[0]
    return proba


def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    # Normalize heatmap to [0,1]
    h = heatmap - np.min(heatmap)
    if h.max() > 0:
        h = h / (h.max() + 1e-8)
    h_rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    # simple colormap: red for high importance
    h_rgb[..., 0] = h  # R
    h_rgb[..., 1] = 0.0  # G
    h_rgb[..., 2] = 1.0 - h  # B
    heat = (h_rgb * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(img.size, Image.BILINEAR)
    return Image.blend(img.convert("RGB"), heat_img, alpha)


@torch.inference_mode()
def explain_image(
    img_path: Path,
    artifacts_dir: Path,
    class_map_path: Path,
    device: torch.device,
    out_dir: Path,
    target: str = "pred",  # "pred" or "true"
    patch: int = 32,
    stride: int = 16,
) -> Path:
    with open(class_map_path) as f:
        cls_map = json.load(f)
    id_to_cls = {v: k for k, v in cls_map.items()}

    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    pls = joblib.load(artifacts_dir / "pls.pkl")
    svm = joblib.load(artifacts_dir / "svm.pkl")
    model, preprocess, _ = build_efficientnet_b0_embedder(device)

    img = Image.open(img_path).convert("RGB")
    base_proba = predict_proba(model, preprocess, device, scaler, pls, svm, img)
    pred_id = int(np.argmax(base_proba))
    pred_name = id_to_cls.get(pred_id, str(pred_id))

    W, H = img.size
    heat = np.zeros((H, W), dtype=np.float32)

    # Work on a downscaled working grid to speed up
    # We'll compute occlusion on the 224x224 center-cropped version to match the embedder's input better
    resized = img.copy().resize((W, H))
    # Slide occlusion patch
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            occluded = resized.copy()
            # gray patch
            for yy in range(y, y + patch):
                for xx in range(x, x + patch):
                    if 0 <= xx < W and 0 <= yy < H:
                        occluded.putpixel((xx, yy), (127, 127, 127))
            proba = predict_proba(model, preprocess, device, scaler, pls, svm, occluded)
            delta = base_proba.copy()
            if target == "pred":
                score_base = base_proba[pred_id]
                score_occ = proba[pred_id]
            else:
                # if true label unknown, fall back to pred
                score_base = base_proba[pred_id]
                score_occ = proba[pred_id]
            drop = max(0.0, score_base - score_occ)
            heat[y : y + patch, x : x + patch] += drop

    # Smooth/normalize heat
    heat = heat / (patch * patch)
    overlay = overlay_heatmap_on_image(img, heat)
    # Slight contrast bump for visibility
    overlay = ImageEnhance.Contrast(overlay).enhance(1.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"occlusion_{img_path.stem}_{pred_name}.png"
    overlay.save(out_path)

    # Side-by-side composite (original | overlay)
    comp = Image.new("RGB", (img.width * 2, img.height))
    comp.paste(img.convert("RGB"), (0, 0))
    comp.paste(overlay, (img.width, 0))
    comp_path = out_dir / f"occlusion_composite_{img_path.stem}_{pred_name}.png"
    comp.save(comp_path)

    # Append mapping CSV: overlay, composite, original, predicted, base proba
    import csv
    map_csv = out_dir / "occlusion_index.csv"
    write_header = not map_csv.exists()
    with map_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["overlay_path", "composite_path", "original_path", "predicted", "proba_0", "proba_1", "proba_2"])
        row = [str(out_path), str(comp_path), str(img_path), pred_name]
        row += [float(x) for x in base_proba]
        w.writerow(row)

    return comp_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="/teamspace/studios/this_studio/artifacts/cnn_fast_cw")
    parser.add_argument("--class_map", type=str, default="/teamspace/studios/this_studio/dataset/metadata/class_map.json")
    parser.add_argument("--images", type=str, nargs="+", help="image paths to explain")
    parser.add_argument("--out_dir", type=str, default="/teamspace/studios/this_studio/artifacts/explain")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_paths = []
    for p in args.images:
        out = explain_image(Path(p), Path(args.artifacts_dir), Path(args.class_map), device, Path(args.out_dir), patch=args.patch, stride=args.stride)
        print(out)


if __name__ == "__main__":
    main()


