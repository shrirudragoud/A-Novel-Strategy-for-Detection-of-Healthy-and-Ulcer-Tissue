import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_root", type=str, default="/teamspace/studios/this_studio/dataset/features_cnn")
    parser.add_argument("--artifacts_dir", type=str, default="/teamspace/studios/this_studio/artifacts/cnn_fast_cw")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--tsne", action="store_true")
    args = parser.parse_args()

    console = Console()
    feat = Path(args.features_root)
    art = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir) if args.out_dir else art
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    X = np.load(feat / f"{args.split}_X.npy")
    y = np.load(feat / f"{args.split}_y.npy")
    paths_file = feat / f"{args.split}_paths.txt"
    rel_paths = paths_file.read_text().splitlines() if paths_file.exists() else ["?"] * len(y)

    scaler = joblib.load(art / "scaler.pkl")
    pls = joblib.load(art / "pls.pkl")
    svm = joblib.load(art / "svm.pkl")

    X2 = scaler.transform(X)
    X3 = pls.transform(X2)
    y_pred = svm.predict(X3)
    try:
        y_proba = svm.predict_proba(X3)
    except Exception:
        y_proba = None

    # Reports
    rep = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()
    (out_dir / f"report_{args.split}.json").write_text(json.dumps({"report": rep, "confusion_matrix": cm}, indent=2))
    console.print(f"Saved report to {(out_dir / f'report_{args.split}.json')} ")

    # Misclassifications CSV
    import csv
    miscsv = out_dir / f"misclassified_{args.split}.csv"
    with miscsv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["path", "y_true", "y_pred"]
        if y_proba is not None:
            header += ["proba_0", "proba_1", "proba_2"]
        w.writerow(header)
        for i, (yt, yp) in enumerate(zip(y, y_pred)):
            if yt != yp:
                row = [rel_paths[i], int(yt), int(yp)]
                if y_proba is not None:
                    row += [float(x) for x in y_proba[i]]
                w.writerow(row)
    console.print(f"Saved misclassifications to {miscsv}")

    # t-SNE plot (optional)
    if args.tsne:
        console.print("Computing t-SNE on PLS space (this may take a while)...")
        Z = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42).fit_transform(X3)
        fig, ax = plt.subplots(figsize=(6,5))
        scatter = ax.scatter(Z[:,0], Z[:,1], c=y, cmap="tab10", s=6, alpha=0.7)
        ax.set_title(f"t-SNE ({args.split}) on PLS space")
        fig.tight_layout()
        fig.savefig(out_dir / f"tsne_{args.split}.png", dpi=150)
        plt.close(fig)
        console.print(f"Saved t-SNE to {out_dir / f'tsne_{args.split}.png'}")


if __name__ == "__main__":
    main()


