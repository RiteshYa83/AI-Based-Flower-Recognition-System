"""Train the flower classifier.

Expected dataset layout (one folder per species):

    data/flower_photos/
        daisy/      img1.jpg img2.jpg ...
        rose/       ...
        sunflower/  ...

Usage:
    python train.py --data-dir data/flower_photos
    python train.py --data-dir data/flower_photos --features mobilenet
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_TYPES, extract_features, read_image_path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(data_dir, feature_type, limit=None):
    root = Path(data_dir)
    if not root.is_dir():
        sys.exit(f"Dataset folder not found: {root}")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        sys.exit("Need at least two class sub-folders inside the dataset folder.")

    X, y = [], []
    for cls in classes:
        files = sorted(f for f in (root / cls).rglob("*") if f.suffix.lower() in EXTS)
        if limit:
            files = files[:limit]
        print(f"  {cls:<15} {len(files)} images")
        for f in files:
            try:
                X.append(extract_features(read_image_path(f), feature_type))
                y.append(cls)
            except Exception as exc:
                print(f"    skipped {f.name}: {exc}")
    return np.array(X), np.array(y)


def candidate_models():
    return {
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=10, gamma="scale", probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/flower_photos")
    ap.add_argument("--features", choices=FEATURE_TYPES, default="classic")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per class (handy for quick tests)")
    args = ap.parse_args()

    print("Extracting features...")
    X, y = load_dataset(args.data_dir, args.features, args.limit)
    print(f"Total: {len(y)} images, {X.shape[1]} features each\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    n_splits = min(5, int(min(np.unique(y_tr, return_counts=True)[1])))
    if n_splits < 2:
        sys.exit("Each class needs at least a few images.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Comparing models ({n_splits}-fold cross-validation on training data):")
    scores = {}
    for name, model in candidate_models().items():
        scores[name] = cross_val_score(model, X_tr, y_tr, cv=cv, n_jobs=-1).mean()
        print(f"  {name:<20} {scores[name]:.3f}")

    best = max(scores, key=scores.get)
    model = candidate_models()[best]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\nBest model: {best}")
    print(f"Held-out test accuracy: {acc:.3f}\n")
    print(classification_report(y_te, y_pred))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_type": args.features},
                out / "flower_model.joblib")

    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, cmap="Blues",
                                            xticks_rotation=45)
    plt.title(f"{best} - test accuracy {acc:.2f}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close()

    (out / "metrics.json").write_text(json.dumps({
        "best_model": best,
        "feature_type": args.features,
        "cv_scores": scores,
        "test_accuracy": acc,
        "classes": sorted(set(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }, indent=2))
    print(f"Saved model and reports to {out}/")


if __name__ == "__main__":
    main()
