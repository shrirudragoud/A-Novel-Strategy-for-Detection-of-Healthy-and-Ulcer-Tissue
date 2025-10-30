## Inference and Analysis Guide

### Inference on new images (CNN + PLS + SVM)
```bash
# After setting PYTHONPATH to repo root
python tools/infer_images.py \
  --artifacts_dir artifacts/cnn_fast_cw \
  --images /path/to/img1.jpg /path/to/img2.png \
  --device auto \
  --out_csv artifacts/infer_results.csv
```
Outputs a CSV with path, predicted class, and probabilities.

### Analyze predictions on a split
```bash
# After setting PYTHONPATH to repo root
python tools/analyze_predictions.py \
  --features_root dataset/features_cnn \
  --artifacts_dir artifacts/cnn_fast_cw \
  --split test --tsne
```
Artifacts:
- `report_test.json` (classification_report + confusion matrix)
- `misclassified_test.csv` (path, y_true, y_pred, probabilities)
- `tsne_test.png` (2D visualization of PLS-space embeddings)

### Notes
- Grad-CAM: since the final classifier is an SVM on CNN embeddings, class-specific Grad-CAM is not directly applicable without a trainable CNN head. If needed, we can add an end-to-end fine-tuned head and then run Grad-CAM.
- SHAP/LIME on embeddings: can be added, but permutation importance on PLS-space features is often more stable and cheaper. We can integrate this on request.


