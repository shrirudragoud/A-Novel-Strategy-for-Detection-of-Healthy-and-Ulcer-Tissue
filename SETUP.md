# Setup Instructions

## 1. Clone the Repository
```bash
git clone https://github.com/shrirudragoud/A-Novel-Strategy-for-Detection-of-Healthy-and-Ulcer-Tissue.git
cd A-Novel-Strategy-for-Detection-of-Healthy-and-Ulcer-Tissue
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Set PYTHONPATH
**Required for imports to work correctly.**
```bash
export PYTHONPATH=$(pwd)
# Or manually: export PYTHONPATH=/absolute/path/to/A-Novel-Strategy-for-Detection-of-Healthy-and-Ulcer-Tissue
```

To make this permanent, add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export PYTHONPATH=/path/to/A-Novel-Strategy-for-Detection-of-Healthy-and-Ulcer-Tissue' >> ~/.bashrc
```

## 4. Dataset Setup

### Option A: Download from Google Drive
1. Download the dataset from the provided Google Drive link
2. Extract images to `dataset/images/` with the following structure:
   ```
   dataset/images/
   ├── healthy/
   │   ├── image1.jpg
   │   └── ...
   ├── ulcer/
   │   ├── image1.jpg
   │   └── ...
   └── polyp/
       ├── image1.jpg
       └── ...
   ```

### Option B: Use Your Own Dataset
Organize your images in the same structure as above.

## 5. Generate Splits and Labels
```bash
python tools/make_splits.py
```

This creates:
- `dataset/splits/{train,val,test}.txt`
- `dataset/metadata/labels.csv`
- `dataset/metadata/class_map.json`

## 6. Extract CNN Embeddings
```bash
python tools/extract_cnn_embeddings.py \
  --dataset_root dataset \
  --out_root dataset/features_cnn \
  --splits train val test --batch_size 64 --device auto
```

## 7. Train Model
```bash
python tools/train_pls_svm.py \
  --features_root dataset/features_cnn \
  --artifacts_dir artifacts \
  --run_name cnn_fast_cw \
  --preset fast \
  --oversample_multipliers 1,6,40 \
  --proba_reweight_grid "1,1,1;1,3,10;1,4,15"
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'src'`, ensure:
1. You're in the repository root directory
2. PYTHONPATH is set correctly: `echo $PYTHONPATH` should show the repo root path

### CUDA/GPU Issues
- Use `--device cpu` if CUDA is unavailable
- The code will auto-detect GPU if available with `--device auto`

