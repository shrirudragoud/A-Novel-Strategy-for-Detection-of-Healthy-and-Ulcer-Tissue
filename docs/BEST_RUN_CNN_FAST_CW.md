## Best Run: CNN Embeddings + PLS + SVM (fast preset, cw + reweight)

### Command used
```bash
# After setting PYTHONPATH to repo root
python tools/train_pls_svm.py \
  --features_root dataset/features_cnn \
  --artifacts_dir artifacts \
  --run_name cnn_fast_cw \
  --preset fast \
  --oversample_multipliers 1,6,40 \
  --proba_reweight_grid "1,1,1;1,3,10;1,4,15"
```

### Dataset sizes
- Train X: (28197, 1280), y: (28197)
- Val   X: (3523, 1280), y: (3523)
- Test  X: (3527, 1280), y: (3527)
- Train distribution: {0: 27470, 1: 683, 2: 44}
- Val distribution:   {0: 3433, 1: 85, 2: 5}
- Test distribution:  {0: 3435, 1: 86, 2: 6}

Oversampling applied: multipliers=[1,6,40] → Train X: (33328, 1280), y: (33328)

### Grid
- Candidates: 4, Folds: 3 → Total fits: 12
- Best params: `pls__n_components=24`, `svm__C=4`, `svm__gamma=scale`, `svm__class_weight=balanced`

### Metrics
- Validation: Accuracy 0.9991, Macro‑F1 0.9939
- Test:       Accuracy 0.9983, Macro‑F1 0.9882

### Artifacts
- `artifacts/cnn_fast_cw/`
  - `scaler.pkl`, `pls.pkl`, `svm.pkl`
  - `metrics.json`
  - `confusion_val.png`, `confusion_test.png`
  - `roc_val.png`, `roc_test.png`

### Notes
- CNN embeddings provide strong separability; oversampling and class weighting improve minority class recall.
- Probability reweighting tested on validation is applied to test when beneficial.


