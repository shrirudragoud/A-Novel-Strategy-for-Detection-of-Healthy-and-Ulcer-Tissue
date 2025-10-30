## Pipeline and Modules (Deep Dive)

### 1) Data ingestion
- `dataset/splits/*.txt`: absolute root is `dataset/`; lines are relative (e.g., `images/healthy/...jpg`).
- `src/data/loader.py`:
  - `read_split_file(root, split)`: returns absolute Paths per split
  - `read_labels_csv(root)`: maps `filepath` → `label_id`
  - `iter_split_samples(...)`: yields `(rgb_uint8, label, rel)` with resize

### 2) Preprocessing (classical path)
- `src/preprocess/image_ops.py`:
  - `to_colorspace` (HSV/YCbCr), `median_denoise`, `extract_luminance`
  - `preprocess_luminance(rgb, colorspace, kernel_size)`

### 3) Features
- IMF features: `src/features/emd_features.py`
  - 1D projections (rows/cols mean/median) → EMD → stats per IMF (mean, std, energy, entropy, skewness, kurtosis)
  - `extract_emd_features(luminance, max_imfs=4, projections=(...))`
- CNN embeddings: `src/deep/cnn_embedder.py`
  - `build_efficientnet_b0_embedder(device)` returns (model, preprocess, emb_dim=1280)
  - `tools/extract_cnn_embeddings.py` saves `{split}_X.npy, {split}_y.npy, {split}_paths.txt`

### 4) Training
- `tools/train_pls_svm.py` (pipeline):
  - Steps: `SimpleImputer` → `StandardScaler` → `PLSTransformer` → `SVC(RBF, probability=True)`
  - Grid-search with StratifiedKFold; options:
    - `--preset {full,fast}`: controls grid/folds
    - `--pls_components`, `--C_list`, `--gamma_list`
    - `--class_weight_grid {none,balanced}` and custom dicts
    - `--oversample_multipliers a,b,c` to duplicate classes in train
    - `--proba_reweight_grid` (post-hoc probability scaling; val-chosen applied to test)
  - Artifacts: `scaler.pkl`, `pls.pkl`, `svm.pkl`, `metrics.json`, `confusion_*.png`, `roc_*.png`

### 5) Inference and analysis
- `tools/infer_images.py`: run CNN+PLS+SVM on arbitrary images; CSV output
- `tools/analyze_predictions.py`: metrics JSON, misclassifications CSV, t‑SNE
- `tools/explain_occlusion.py`: occlusion maps (overlay + composite + index CSV)
- `tools/build_explain_gallery.py`: HTML gallery for explain outputs

### 6) Imbalance handling
- Class weighting (`svm__class_weight='balanced'`)
- Oversampling via `--oversample_multipliers`
- Post-hoc probability reweighting via `--proba_reweight_grid`

### 7) Versioning and reproducibility
- Each run writes to `artifacts/<run_name>/`.
- Seeded randomness in selection/oversampling where relevant.


