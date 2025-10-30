## 3-Class Endoscopy Classifier: Practical Plan (Senior ML Design)

### 1) Objective
- Classify endoscopy images into three classes: `healthy`, `ulcer`, `polyp`.
- Optimize for strong accuracy with a simple, maintainable classical ML stack.

### 2) Current Data Status
- Images organized under `dataset/images/{healthy,ulcer,polyp}` ✓
- `dataset/metadata/class_map.json` with mapping `{healthy:0, ulcer:1, polyp:2}` ✓
- `dataset/metadata/labels.csv` populated ✓
- Stratified splits in `dataset/splits/{train,val,test}.txt` ✓

This satisfies data readiness for feature extraction and model training.

### 3) End-to-End Pipeline (High Level)
1. Load images per split using file lists.
2. Preprocess: resize → color-space (HSV or YCbCr) → denoise.
3. Extract IMF-based features via EMD/TV-EMD; compute summary stats per IMF.
4. Feature scaling (StandardScaler) on train only.
5. Dimensionality reduction with PLS on train only; transform val/test.
6. Train RBF-SVM on train; tune hyperparameters via CV; select best.
7. Evaluate on val/test; report accuracy, macro F1, confusion matrix.
8. Persist artifacts (`scaler`, `pls`, `svm`, metrics report).

### 4) Detailed Steps and Rationale

4.1 Image Loading
- Use split files in `dataset/splits/` to build lists of image paths.
- Labels from `labels.csv` keyed by relative path.

4.2 Preprocessing
- Resize: 128×128 (fast) or 224×224 (more feature detail). Start with 128×128.
- Color transform: prefer HSV or YCbCr; use the luminance channel for EMD to reduce compute.
- Denoise: median filter (3×3) to reduce speckle while preserving edges.

4.3 IMF Feature Extraction (EMD/TV-EMD)
- Decompose the luminance channel into K IMFs (target K≈3–5). If variable, keep first K by energy; if fewer IMFs, zero-pad stats.
- For each IMF compute: mean, std, energy, entropy, skewness, kurtosis.
- Concatenate stats across IMFs → feature vector per image.
- Libraries: `PyEMD` for EMD; TV-EMD variants optional (slower but more stable). If EMD is too slow, fallback to texture features (LBP/Gabor) as a plan B.

4.4 Feature Scaling
- Standardize features with `StandardScaler` fitted on train only; apply to val/test.

4.5 Partial Least Squares (PLS)
- Use `PLSRegression` with one-hot labels; tune `n_components` (e.g., 8–32). Choose via cross-validation on the training set.
- Transform val/test with the fitted PLS.

4.6 Classifier: SVM with RBF (Recommended First)
- `SVC(kernel='rbf')`; tune `C` and `gamma` via grid-search or Bayesian search.
- Use class weights if imbalance is noticeable (`class_weight='balanced'`).
- Alternative: custom RBF neural layer is possible but adds complexity; use later if needed.

4.7 Evaluation
- Metrics: overall accuracy, macro F1, per-class F1, confusion matrix.
- Optionally plot per-class ROC (one-vs-rest) if probabilistic outputs are used.
- Always evaluate on the untouched test split after model selection.

4.8 Persistence and Reproducibility
- Save: `scaler.pkl`, `pls.pkl`, `svm.pkl`, and a JSON `metrics.json`.
- Fix random seeds; log versions; keep a `requirements.txt`.

### 5) Implementation Notes
- Performance: EMD is the most expensive step. To keep it tractable:
  - Use only luminance (HSV-V or YCbCr-Y) for EMD.
  - Cap number of IMFs to first 3–4 components.
  - Batch feature extraction with multiprocessing.
- Consistency: Ensure a fixed-length feature vector per image regardless of EMD variability.
- Leakage avoidance: fit scaler and PLS only on train; transform val/test separately.

### 6) Baselines and Sanity Checks
- Baseline A: Simple RGB histogram or HOG + linear SVM. Should underperform but provides a sanity benchmark.
- Baseline B: CNN embedding (e.g., EfficientNetB0) + SVM. Useful fallback if EMD becomes a bottleneck.

### 7) Directory Structure (Working)
- `dataset/images/{healthy,ulcer,polyp}`: input images
- `dataset/metadata/{class_map.json,labels.csv}`: label mapping and index
- `dataset/splits/{train,val,test}.txt`: file lists
- `artifacts/`: saved scaler, PLS, SVM, metrics
- `tools/`: scripts (e.g., `make_splits.py`)
- `src/`: pipeline code (loader, preprocessing, features, model, eval)

### 8) Minimal Build Plan (Next Steps to Implement)
1. Data loader: reads split files, yields `(image_array, label_id)`.
2. Preprocessing: resize → HSV/YCbCr → median denoise → luminance extraction.
3. Feature extractor: EMD on luminance → IMF stats vector.
4. Fit stack: scaler → PLS (n_components search) → SVM (C, gamma search).
5. Evaluation script: metrics + confusion matrix; save artifacts.

### 9) Risks and Mitigations
- EMD runtime: mitigate via luminance-only and IMF cap; parallelize.
- Class imbalance: monitor per-class counts; use class weights.
- Overfitting in PLS: tune `n_components`; use validation; prefer macro F1.

### 10) Success Criteria
- Clear improvement over a simple baseline (e.g., +10–15 pts in macro F1).
- Stable metrics across multiple random seeds.
- Reproducible training with saved artifacts and documented versions.


