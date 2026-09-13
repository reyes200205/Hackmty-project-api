"""Entrena un clasificador de voz sintetica calibrado para CLIPS CORTOS
(segmentos de 1.5-15s), usando los ~2900 segmentos reales extraidos del
dataset del reto (build_voice_dataset.py). Split train/val ya viene dado
por el manifest (disjunto por hablante).

Uso: python scripts/train_voice_classifier.py
Genera: bank/models/voice_short_clip_classifier.joblib
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.features import FEATURE_NAMES  # noqa: E402

CSV_PATH = Path(__file__).parent / "voice_dataset.csv"
MODEL_OUT = PROJECT_ROOT / "bank" / "models" / "voice_short_clip_classifier.joblib"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"{len(df)} segmentos totales")
    print(df["label"].value_counts())
    print(df["split"].value_counts())

    X = df[FEATURE_NAMES].values
    y = (df["label"] == "synthetic").astype(int).values

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"\ntrain: {len(X_train)} segmentos, val: {len(X_val)} segmentos")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        acc = accuracy_score(y_val, val_pred)
        auc = roc_auc_score(y_val, val_proba)
        print(f"\n=== {name} ===")
        print(f"val accuracy={acc:.4f} auc={auc:.4f}")
        print(classification_report(y_val, val_pred, target_names=["human", "synthetic"]))
        results[name] = (model, acc, auc)

    # Se prioriza accuracy (con AUC tan cerrado entre modelos, mas errores
    # netos en produccion importan mas que el ranking de probabilidades).
    best_name = max(results, key=lambda n: results[n][1])
    best_model, best_acc, best_auc = results[best_name]
    print(f"\nMejor modelo: {best_name} (val accuracy={best_acc:.4f}, auc={best_auc:.4f})")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler,
        "model": best_model,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "val_auc": best_auc,
    }, MODEL_OUT)
    print(f"Guardado en {MODEL_OUT}")


if __name__ == "__main__":
    main()
