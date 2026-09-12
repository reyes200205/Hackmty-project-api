from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from detector.conversational import extract_conversational_features

HACKMTY_DIR = Path("../hackmty26")
MANIFEST_PATH = HACKMTY_DIR / "manifest.csv"
AUDIO_DIR = HACKMTY_DIR / "audio"
MODEL_DIR = Path("detector")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "conversational_model.joblib"

FEATURE_KEYS = [
    "pos_latency_mean",
    "pos_latency_std",
    "latency_mean",
    "caller_turn_dur_mean",
    "caller_turn_dur_std",
    "overlap_duration",
    "interruptions_by_caller",
    "interruptions_by_agent",
    "long_silence_reaction_ratio",
    "caller_turn_count",
]


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    data_rows = []

    for _, row in manifest.iterrows():
        wav_file = AUDIO_DIR / f"{row['anon_id']}.wav"
        if not wav_file.exists():
            continue

        data_wav, sample_rate = sf.read(wav_file, dtype="float32", always_2d=True)
        caller, agent = data_wav[:, 0], data_wav[:, 1]

        features = extract_conversational_features(caller, agent, sample_rate)
        features["anon_id"] = row["anon_id"]
        features["label"] = row["label"]
        features["split"] = row["split"]
        data_rows.append(features)

    df = pd.DataFrame(data_rows)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    X_train = train_df[FEATURE_KEYS].values
    y_train = (train_df["label"] == "synthetic").astype(int).values

    X_val = val_df[FEATURE_KEYS].values
    y_val = (val_df["label"] == "synthetic").astype(int).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Entrenar clasificador probabilistico calibrado
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    val_prob = clf.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_prob)

    print(f"Modelo Conversacional Entrenado Exitosamente:")
    print(f"  Accuracy en Val: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC en Val:  {auc:.4f}")
    print("\nReporte de Clasificación (Val):")
    print(classification_report(y_val, val_pred, target_names=["Humano", "Sintético"]))

    # Guardar artefacto del modelo
    model_artifact = {
        "scaler": scaler,
        "classifier": clf,
        "feature_keys": FEATURE_KEYS,
        "metrics": {"val_acc": float(acc), "val_auc": float(auc)},
    }
    joblib.dump(model_artifact, MODEL_FILE)
    print(f"Artefacto guardado en: {MODEL_FILE}")


if __name__ == "__main__":
    main()
