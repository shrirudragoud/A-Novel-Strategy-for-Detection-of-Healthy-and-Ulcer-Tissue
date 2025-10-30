## Codebase Overview

### Directory structure (key parts)
- `dataset/`
  - `images/{healthy,ulcer,polyp}`: raw images
  - `splits/{train,val,test}.txt`: relative paths per split
  - `metadata/{class_map.json,labels.csv}`: label mapping and index
  - `features/`: classical IMF feature arrays (optional)
  - `features_cnn/`: CNN embeddings `{split}_X.npy, {split}_y.npy, {split}_paths.txt`
- `artifacts/`
  - Model artifacts per run: `scaler.pkl, pls.pkl, svm.pkl, metrics.json, confusion_*.png, roc_*.png`
  - Explainability: `explain/` contains occlusion overlays, composites, index
- `src/`
  - `data/loader.py`: split readers, label loading, image loading
  - `preprocess/image_ops.py`: HSV/YCbCr conversion, denoise, luminance
  - `features/emd_features.py`: EMD IMF features and stats
  - `models/pls_transformer.py`: supervised PLSTransformer wrapper
  - `deep/cnn_embedder.py`: EfficientNetB0 embedding builder
- `tools/`
  - `make_splits.py`: build `labels.csv` and `splits/*.txt`
  - `preview_preprocessing.py`: sanity check preprocessing
  - `extract_features.py`: build IMF-based features
  - `extract_cnn_embeddings.py`: build CNN embeddings
  - `train_pls_svm.py`: training pipeline (scaler→PLS→SVM) with grid search, plots, logging
  - `analyze_predictions.py`: reports, misclassifications CSV, t‑SNE
  - `infer_images.py`: run inference on new images with CNN+PLS+SVM
  - `explain_occlusion.py`: occlusion-based confidence maps (overlay/composite + index CSV)
  - `build_explain_gallery.py`: HTML gallery for explain outputs

### Data flow at a glance
1) Data → `dataset/images/...`, `splits/*.txt`, `metadata/*`
2) Feature extraction (either IMF or CNN) → `dataset/features*/*.npy`
3) Training → `artifacts/<run_name>/*` (pickled components, metrics, plots)
4) Inference/analysis/explain → tools consume `artifacts/*` + inputs and produce results

### Key decisions
- Use classical ML pipeline for tabular features (scaler→PLS→SVM) with CV.
- Use CNN embeddings (EfficientNetB0) for better separability; keep the classical head for simplicity.
- Address imbalance by oversampling/weights/reweighting, with telemetry and versioned runs.


