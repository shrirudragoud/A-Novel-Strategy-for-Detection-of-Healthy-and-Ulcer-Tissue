## Commands Reference

**Important**: Set PYTHONPATH to the root of the cloned repository on your machine.

```bash
# After cloning, navigate to the repo root and run:
export PYTHONPATH=$(pwd)
# Or manually: export PYTHONPATH=/path/to/A-Novel-Strategy-for-Detection-of-Healthy-and-Ulcer-Tissue
```

All commands below assume you've set PYTHONPATH and are in the repo root directory.

### Dataset setup
- Create splits and labels
```bash
python tools/make_splits.py
```

### Classical features (optional)
- Extract IMF features
```bash
python tools/extract_features.py \
  --dataset_root dataset \
  --colorspace HSV --height 128 --width 128 --max_imfs 4 \
  --projections rows_mean,cols_mean
```

### CNN embeddings
- Extract EfficientNetB0 embeddings
```bash
python tools/extract_cnn_embeddings.py \
  --dataset_root dataset \
  --out_root dataset/features_cnn \
  --splits train val test --batch_size 64 --device auto
```

### Training (PLS + SVM)
- Fast preset with oversampling and probability reweighting
```bash
python tools/train_pls_svm.py \
  --features_root dataset/features_cnn \
  --artifacts_dir artifacts \
  --run_name cnn_fast_cw \
  --preset fast \
  --oversample_multipliers 1,6,40 \
  --proba_reweight_grid "1,1,1;1,3,10;1,4,15"
```

### Analysis & inference
- Analyze split predictions (report, misclass CSV, t‑SNE)
```bash
python tools/analyze_predictions.py \
  --features_root dataset/features_cnn \
  --artifacts_dir artifacts/cnn_fast_cw \
  --split test --tsne
```
- Inference on new images
```bash
python tools/infer_images.py \
  --artifacts_dir artifacts/cnn_fast_cw \
  --images /path/to/img1.jpg /path/to/img2.png \
  --device auto \
  --out_csv artifacts/infer_results.csv
```
- Explainability (occlusion maps)
```bash
python tools/explain_occlusion.py \
  --artifacts_dir artifacts/cnn_fast_cw \
  --class_map dataset/metadata/class_map.json \
  --images dataset/images/ulcer/example.jpg \
  --patch 16 --stride 8 --device auto \
  --out_dir artifacts/explain
```
- Build explain gallery
```bash
python tools/build_explain_gallery.py \
  --explain_dir artifacts/explain
```

### Binary (Healthy vs Ulcer) balanced test run (separate folder)
- Create balanced features in `features_cnn_bin_test/`
```bash
python - << 'PY'
import numpy as np, pathlib as p, random
random.seed(42)
src=p.Path("dataset/features_cnn")
dst=p.Path("dataset/features_cnn_bin_test"); dst.mkdir(parents=True, exist_ok=True)
for split in ["train","val","test"]:
    X=np.load(src/f"{split}_X.npy"); y=np.load(src/f"{split}_y.npy")
    keep=np.where(np.isin(y,[0,1]))[0]; X,y=X[keep],y[keep]
    if split=="train":
        i_h=np.where(y==0)[0].tolist(); i_u=np.where(y==1)[0].tolist(); random.shuffle(i_h)
        i_h=i_h[:len(i_u)]; idx=np.array(i_h+i_u); np.random.shuffle(idx); X,y=X[idx],y[idx]
    np.save(dst/f"{split}_X.npy",X); np.save(dst/f"{split}_y.npy",y)
    print(split,X.shape,y.shape)
PY
```
- Train on balanced binary features
```bash
python tools/train_pls_svm.py \
  --features_root dataset/features_cnn_bin_test \
  --artifacts_dir artifacts \
  --run_name cnn_bin_fast_test \
  --preset fast \
  --proba_reweight_grid ""
```


