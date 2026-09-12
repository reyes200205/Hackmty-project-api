"""A4: Entrena y guarda el calibrador isotónico para el ensamble acústico + conversacional.

Se ajusta sobre predicciones out-of-fold (validación cruzada) del split de train, no sobre
el split de val: el ensamble ya clasifica val al 100%, sin ningún caso dudoso ni error, así
que calibrar directo contra val produce una curva en escalón (0.01 o 0.99 siempre, sin
gradación real) que generaliza muy mal en el set oculto del juez, donde sí habrá casos
genuinamente difíciles. Usando out-of-fold sobre train, cada predicción viene de un modelo
que nunca vio esa llamada, así que sí aparecen errores y casos al filo para calibrar de verdad.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from detector.features import feature_vector
from detector.conversational import extract_conversational_features, predict_conversational
from detector.inference import ACOUSTIC_WEIGHT, _acoustic_confidence

HACKMTY_DIR = Path(__file__).resolve().parent.parent.parent / "hackmty26"
AUDIO_DIR = HACKMTY_DIR / "audio"
MANIFEST_PATH = HACKMTY_DIR / "manifest.csv"
CALIBRATOR_PATH = Path(__file__).parent / "model" / "calibrator.joblib"

CONV_FEATURE_KEYS = [
    "pos_latency_mean", "pos_latency_std", "latency_mean",
    "caller_turn_dur_mean", "caller_turn_dur_std", "overlap_duration",
    "interruptions_by_caller", "interruptions_by_agent",
    "long_silence_reaction_ratio", "caller_turn_count",
]


def _load_raw_channels(anon_id: str):
    data, sr = sf.read(str(AUDIO_DIR / f"{anon_id}.wav"), dtype="float32", always_2d=True)
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    return caller, agent, sr


def _load_features(rows):
    acoustic_X, conv_X, y = [], [], []
    for row in rows:
        caller, agent, sr = _load_raw_channels(row["anon_id"])
        acoustic_X.append(feature_vector(caller, sr))
        conv_feats = extract_conversational_features(caller, agent, sr)
        conv_X.append([conv_feats.get(k, 0.0) for k in CONV_FEATURE_KEYS])
        y.append(1 if row["label"] == "synthetic" else 0)
    return np.array(acoustic_X), np.array(conv_X), np.array(y)


def _acoustic_model_params():
    return dict(
        n_estimators=300, num_leaves=15, min_child_samples=10,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbosity=-1,
    )


def train_calibrator(n_splits: int = 5, seed: int = 42) -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    train_rows = manifest[manifest["split"] == "train"].to_dict("records")
    val_rows = manifest[manifest["split"] == "val"].to_dict("records")

    print(f"Extrayendo features de {len(train_rows)} llamadas de train (para CV)...")
    acoustic_X, conv_X, y = _load_features(train_rows)

    oof_raw = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(acoustic_X, y), 1):
        print(f"  fold {fold}/{n_splits}: entrenando en {len(fit_idx)}, prediciendo {len(hold_idx)} fuera de muestra...")

        a_scaler = StandardScaler().fit(acoustic_X[fit_idx])
        a_clf = lgb.LGBMClassifier(**_acoustic_model_params())
        a_clf.fit(a_scaler.transform(acoustic_X[fit_idx]), y[fit_idx])

        c_scaler = StandardScaler().fit(conv_X[fit_idx])
        c_clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
        c_clf.fit(c_scaler.transform(conv_X[fit_idx]), y[fit_idx])

        a_p = a_clf.predict_proba(a_scaler.transform(acoustic_X[hold_idx]))[:, 1]
        c_p = c_clf.predict_proba(c_scaler.transform(conv_X[hold_idx]))[:, 1]
        oof_raw[hold_idx] = ACOUSTIC_WEIGHT * a_p + (1 - ACOUSTIC_WEIGHT) * c_p

    n_errors = int(np.sum((oof_raw >= 0.5).astype(int) != y))
    print(f"\nPredicciones out-of-fold: {n_errors} errores de {len(y)} (esto es lo que calibra la curva)")

    calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    calibrator.fit(oof_raw, y)

    brier_before = brier_score_loss(y, oof_raw)
    brier_after = brier_score_loss(y, calibrator.predict(oof_raw))
    print(f"Brier out-of-fold  antes: {brier_before:.4f}  despues: {brier_after:.4f}")

    print(f"\nValidando contra {len(val_rows)} llamadas de val con los modelos finales (los que se despliegan)...")
    val_raw, val_y = [], []
    for row in val_rows:
        caller, agent, sr = _load_raw_channels(row["anon_id"])
        conv_synth, conv_conf, _ = predict_conversational(caller, agent, sr)
        conv_p = conv_conf if conv_synth else (1.0 - conv_conf)
        acoust_p = _acoustic_confidence(caller, sr)
        val_raw.append(ACOUSTIC_WEIGHT * acoust_p + (1 - ACOUSTIC_WEIGHT) * conv_p)
        val_y.append(1 if row["label"] == "synthetic" else 0)
    val_raw = np.array(val_raw)
    val_y = np.array(val_y)
    val_calibrated = calibrator.predict(val_raw)

    print(f"Val accuracy:        {accuracy_score(val_y, (val_calibrated >= 0.5).astype(int)):.4f}")
    print(f"Val Brier crudo:     {brier_score_loss(val_y, val_raw):.4f}")
    print(f"Val Brier calibrado: {brier_score_loss(val_y, val_calibrated):.4f}")
    print("Muestra de valores calibrados en val (deberian variar, no ser solo 0.02/0.98):")
    print(np.round(val_calibrated[:15], 3))

    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrator": calibrator, "brier_score": float(brier_after)}, CALIBRATOR_PATH)
    print(f"\nCalibrador guardado en: {CALIBRATOR_PATH}")


if __name__ == "__main__":
    train_calibrator()
