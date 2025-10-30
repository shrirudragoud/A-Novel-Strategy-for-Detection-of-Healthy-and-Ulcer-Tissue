import argparse
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.models.pls_transformer import PLSTransformer
import joblib
from rich.console import Console
from rich.table import Table


def load_split_features(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(root / f"{split}_X.npy")
    y = np.load(root / f"{split}_y.npy")
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_root", type=str, default="/teamspace/studios/this_studio/dataset/features")
    parser.add_argument("--artifacts_dir", type=str, default="/teamspace/studios/this_studio/artifacts")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--preset", type=str, default="full", choices=["full", "fast"], help="fast reduces grid size for quicker iterations")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--pls_components", type=int, nargs="*", default=[8, 12, 16, 24, 32])
    parser.add_argument("--C_list", type=float, nargs="*", default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--gamma_list", type=str, nargs="*", default=["scale", "auto"])  # can add floats
    parser.add_argument("--class_weight_grid", type=str, nargs="*", default=["none", "balanced"], choices=["none", "balanced"]) 
    parser.add_argument("--class_weight_dicts", type=str, default="", help="Semicolon-separated dicts as weights per class id, e.g. '1,10,30;1,6,20' -> {0:1,1:10,2:30} and {0:1,1:6,2:20}")
    parser.add_argument("--proba_reweight_grid", type=str, default="1,1,1;1,3,10", help="Semicolon-separated class probability multipliers tried post-hoc on val, e.g. '1,1,1;1,3,10'")
    parser.add_argument("--oversample_multipliers", type=str, default="", help="Optional per-class multipliers for train oversampling, e.g. '1,6,40' (for classes 0,1,2)")
    args = parser.parse_args()

    console = Console()
    t0 = time.time()
    features_root = Path(args.features_root)
    base_artifacts_dir = Path(args.artifacts_dir)
    base_artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    artifacts_dir = base_artifacts_dir / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Loading features...[/bold cyan]")
    X_train, y_train = load_split_features(features_root, "train")
    X_val, y_val = load_split_features(features_root, "val")
    X_test, y_test = load_split_features(features_root, "test")

    console.print(f"Train X: {X_train.shape}, y: {y_train.shape}")
    console.print(f"Val   X: {X_val.shape}, y: {y_val.shape}")
    console.print(f"Test  X: {X_test.shape}, y: {y_test.shape}")
    # Class distribution
    def dist(y: np.ndarray) -> dict[int, int]:
        return {int(k): int((y == k).sum()) for k in np.unique(y)}
    console.print(f"[dim]Train dist:[/dim] {dist(y_train)}")
    console.print(f"[dim]Val   dist:[/dim] {dist(y_val)}")
    console.print(f"[dim]Test  dist:[/dim] {dist(y_test)}")

    # Try to load class names
    dataset_root = features_root.parent  # .../dataset
    class_names = None
    try:
        with (dataset_root / "metadata" / "class_map.json").open() as f:
            cls_map = json.load(f)
        id_to_name = {v: k for k, v in cls_map.items()}
        # ensure order by sorted id
        ordered = [id_to_name[i] for i in sorted(id_to_name.keys())]
        class_names = ordered
    except Exception:
        class_names = [str(c) for c in sorted(np.unique(y_train))]

    # Build pipeline: scaler -> PLS -> SVM
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("pls", PLSTransformer(n_components=8)),  # n_components will be tuned
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    # Optionally override to a smaller grid when preset==fast
    if args.preset == "fast":
        # Only override if user didn't explicitly pass custom lists (we assume defaults)
        if args.pls_components == [8, 12, 16, 24, 32]:
            args.pls_components = [16, 24]
        if args.C_list == [0.5, 1, 2, 4, 8]:
            args.C_list = [1, 4]
        if args.gamma_list == ["scale", "auto"]:
            args.gamma_list = ["scale"]
        if args.class_weight_grid == ["none", "balanced"]:
            args.class_weight_grid = ["balanced"]
        # If user didn't set cv_folds, reduce to 3
        if parser.get_default("cv_folds") == args.cv_folds:
            args.cv_folds = 3

    cw_options = [None if x == "none" else "balanced" for x in args.class_weight_grid]
    # Parse explicit dict class weights if provided
    if args.class_weight_dicts.strip():
        for spec in args.class_weight_dicts.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            parts = [float(s) for s in spec.split(",")]
            if len(parts) != 3:
                continue
            d = {0: parts[0], 1: parts[1], 2: parts[2]}
            cw_options.append(d)
    param_grid = {
        "pls__n_components": args.pls_components,
        "svm__C": args.C_list,
        "svm__gamma": args.gamma_list,
        "svm__class_weight": cw_options,
    }

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    # Optional: oversample train set
    if args.oversample_multipliers.strip():
        mults = [int(float(x)) for x in args.oversample_multipliers.split(",")]
        if len(mults) == 3:
            idxs = []
            for cls, m in enumerate(mults):
                cls_idx = np.where(y_train == cls)[0]
                if m > 1 and cls_idx.size > 0:
                    # repeat with replacement to reach approx multiplier
                    reps = m - 1
                    extra = np.random.choice(cls_idx, size=cls_idx.size * reps, replace=True)
                    idxs.append(extra)
            if idxs:
                extra_idx = np.concatenate(idxs)
                X_train = np.concatenate([X_train, X_train[extra_idx]], axis=0)
                y_train = np.concatenate([y_train, y_train[extra_idx]], axis=0)
                console.print(f"[yellow]Applied oversampling:[/yellow] multipliers={mults} -> Train X: {X_train.shape}, y: {y_train.shape}")

    console.print("[bold cyan]Starting grid search (PLS + SVM)...[/bold cyan]")
    total_candidates = (
        len(param_grid["pls__n_components"]) * len(param_grid["svm__C"]) * len(param_grid["svm__gamma"]) * len(param_grid["svm__class_weight"]) 
    )
    console.print(f"[dim]Candidates:[/dim] {total_candidates}  Folds: {args.cv_folds}  Total fits: {total_candidates * args.cv_folds}")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    console.print(f"[green]Best params:[/green] {grid.best_params_}")
    best = grid.best_estimator_

    # Validate on val
    yv_pred = best.predict(X_val)
    if hasattr(best.named_steps["svm"], "predict_proba"):
        yv_proba = best.predict_proba(X_val)
    else:
        yv_proba = None
    val_metrics = {
        "accuracy": float(accuracy_score(y_val, yv_pred)),
        "macro_f1": float(f1_score(y_val, yv_pred, average="macro")),
        "report": classification_report(y_val, yv_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_val, yv_pred).tolist(),
        "best_params": grid.best_params_,
    }
    console.print("[bold cyan]Validation metrics:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Accuracy", f"{val_metrics['accuracy']:.4f}")
    table.add_row("Macro F1", f"{val_metrics['macro_f1']:.4f}")
    console.print(table)

    # Test
    yt_pred = best.predict(X_test)
    if hasattr(best.named_steps["svm"], "predict_proba"):
        yt_proba = best.predict_proba(X_test)
    else:
        yt_proba = None
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, yt_pred)),
        "macro_f1": float(f1_score(y_test, yt_pred, average="macro")),
        "report": classification_report(y_test, yt_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, yt_pred).tolist(),
    }
    console.print("[bold cyan]Test metrics:[/bold cyan]")
    table2 = Table(show_header=True, header_style="bold magenta")
    table2.add_column("Metric")
    table2.add_column("Value")
    table2.add_row("Accuracy", f"{test_metrics['accuracy']:.4f}")
    table2.add_row("Macro F1", f"{test_metrics['macro_f1']:.4f}")
    console.print(table2)

    # Persist artifacts
    joblib.dump(best.named_steps["scaler"], artifacts_dir / "scaler.pkl")
    joblib.dump(best.named_steps["pls"], artifacts_dir / "pls.pkl")
    joblib.dump(best.named_steps["svm"], artifacts_dir / "svm.pkl")
    results_blob = {"val": val_metrics, "test": test_metrics, "cv_best_params": grid.best_params_}
    
    # Post-hoc probability reweighting to improve minority recall
    def parse_reweights(spec: str) -> list[list[float]]:
        out = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            vals = [float(x) for x in part.split(",")]
            if len(vals) == 3:
                out.append(vals)
        return out

    def apply_reweight(proba: np.ndarray, weights: list[float]) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        scaled = proba * w[None, :]
        scaled_sum = scaled.sum(axis=1, keepdims=True)
        # avoid div by zero; no need to renormalize for argmax, but keep stable
        scaled_sum[scaled_sum == 0] = 1.0
        return scaled / scaled_sum

    best_rew = None
    if yv_proba is not None:
        reweights = parse_reweights(args.proba_reweight_grid)
        best_f1 = val_metrics["macro_f1"]
        for rw in reweights:
            adj = apply_reweight(yv_proba, rw)
            yv_pred_adj = np.argmax(adj, axis=1)
            f1_adj = float(f1_score(y_val, yv_pred_adj, average="macro"))
            if f1_adj > best_f1:
                best_f1 = f1_adj
                best_rew = rw
        if best_rew is not None:
            console.print(f"[yellow]Applied proba reweight (val-chosen):[/yellow] {best_rew}  -> macroF1={best_f1:.4f}")
            # Update val metrics
            yv_pred = np.argmax(apply_reweight(yv_proba, best_rew), axis=1)
            val_metrics = {
                "accuracy": float(accuracy_score(y_val, yv_pred)),
                "macro_f1": float(f1_score(y_val, yv_pred, average="macro")),
                "report": classification_report(y_val, yv_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_val, yv_pred).tolist(),
                "best_params": grid.best_params_,
                "proba_reweight": best_rew,
            }
            # Update test with same reweight
            if yt_proba is not None:
                yt_pred = np.argmax(apply_reweight(yt_proba, best_rew), axis=1)
                test_metrics = {
                    "accuracy": float(accuracy_score(y_test, yt_pred)),
                    "macro_f1": float(f1_score(y_test, yt_pred, average="macro")),
                    "report": classification_report(y_test, yt_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, yt_pred).tolist(),
                    "proba_reweight": best_rew,
                }
    results_blob["val"] = val_metrics
    results_blob["test"] = test_metrics
    results_blob["cv_best_params"] = grid.best_params_
    with (artifacts_dir / "metrics.json").open("w") as f:
        json.dump(results_blob, f, indent=2)

    # Save raw predictions/probabilities
    np.save(artifacts_dir / "val_y_true.npy", y_val)
    np.save(artifacts_dir / "val_y_pred.npy", yv_pred)
    if yv_proba is not None:
        np.save(artifacts_dir / "val_y_proba.npy", yv_proba)
    np.save(artifacts_dir / "test_y_true.npy", y_test)
    np.save(artifacts_dir / "test_y_pred.npy", yt_pred)
    if yt_proba is not None:
        np.save(artifacts_dir / "test_y_proba.npy", yt_proba)

    # Plot confusion matrices
    def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label', title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # annotate
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    plot_confusion(np.array(val_metrics["confusion_matrix"]), class_names, "Confusion Matrix (Val)", artifacts_dir / "confusion_val.png")
    plot_confusion(np.array(test_metrics["confusion_matrix"]), class_names, "Confusion Matrix (Test)", artifacts_dir / "confusion_test.png")

    # Plot ROC (one-vs-rest) if probabilities available
    def plot_roc_ovr(y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
        n_classes = y_proba.shape[1]
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
            ax.plot(fpr, tpr, label=f"{labels[i]} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    if yv_proba is not None:
        plot_roc_ovr(y_val, yv_proba, class_names, "ROC (Val)", artifacts_dir / "roc_val.png")
    if yt_proba is not None:
        plot_roc_ovr(y_test, yt_proba, class_names, "ROC (Test)", artifacts_dir / "roc_test.png")

    console.print("[bold green]Saved artifacts:[/bold green]")
    console.print(artifacts_dir / "scaler.pkl")
    console.print(artifacts_dir / "pls.pkl")
    console.print(artifacts_dir / "svm.pkl")
    console.print(artifacts_dir / "metrics.json")
    console.print(artifacts_dir / "confusion_val.png")
    console.print(artifacts_dir / "confusion_test.png")
    if (artifacts_dir / "roc_val.png").exists():
        console.print(artifacts_dir / "roc_val.png")
    if (artifacts_dir / "roc_test.png").exists():
        console.print(artifacts_dir / "roc_test.png")
    console.print(f"[bold blue]Total time:[/bold blue] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


