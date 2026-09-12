"""A2: entrena el clasificador humano/sintetico sobre el dataset de Altur.

Uso:
    python -m detector.train --dataset-dir "C:/ruta/a/hackmty26"

Lee manifest.csv (anon_id,label,split,duration_s) y audio/<anon_id>.wav del
dataset, extrae features acusticas (detector/features.py) del canal del
caller, entrena sobre split=train, evalua sobre split=val, y guarda el
modelo en detector/model/classifier.joblib.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from detector.features import FEATURE_NAMES, feature_vector

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_LIGHTGBM = False

MODEL_PATH = Path(__file__).parent / "model" / "classifier.joblib"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "hackmty26"


def load_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    with open(dataset_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_dataset(dataset_dir: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    audio_dir = dataset_dir / "audio"
    features, labels, ids = [], [], []
    for row in rows:
        wav_path = audio_dir / f"{row['anon_id']}.wav"
        if not wav_path.exists():
            print(f"[skip] no existe {wav_path}", file=sys.stderr)
            continue
        data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        caller = data[:, 0]
        features.append(feature_vector(caller, sample_rate))
        labels.append(1 if row["label"] == "synthetic" else 0)
        ids.append(row["anon_id"])
    return np.vstack(features), np.array(labels), ids


def train(dataset_dir: Path) -> None:
    rows = load_manifest(dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"manifest: train={len(train_rows)} val={len(val_rows)}")

    X_train, y_train, _ = build_dataset(dataset_dir, train_rows)
    X_val, y_val, val_ids = build_dataset(dataset_dir, val_rows)
    print(f"cargados: train={len(y_train)} val={len(y_val)} features={X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if _HAS_LIGHTGBM:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=15,
            min_child_samples=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
        clf.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_train_s, y_train)

    val_proba = clf.predict_proba(X_val_s)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    print("\n=== Validacion ===")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    try:
        print(f"ROC-AUC:  {roc_auc_score(y_val, val_proba):.4f}")
    except ValueError:
        pass
    print(classification_report(y_val, val_pred, target_names=["human", "synthetic"]))

    errors = [(vid, p) for vid, y, p in zip(val_ids, y_val, val_pred) if y != p]
    if errors:
        print(f"Errores en val ({len(errors)}): {errors}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    train(args.dataset_dir.resolve())


if __name__ == "__main__":
    main()
