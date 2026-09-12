import json
from pathlib import Path
import numpy as np
import pandas as pd

HACKMTY_DIR = Path("../hackmty26")
MANIFEST_PATH = HACKMTY_DIR / "manifest.csv"
TURNS_DIR = HACKMTY_DIR / "turns"


def analyze_turns(turns: list[dict]) -> dict:
    """Extrae metricas conversacionales a partir de la lista de turnos."""
    if not turns:
        return {}

    agent_turns = [t for t in turns if t["channel"] == 1]
    caller_turns = [t for t in turns if t["channel"] == 0]

    agent_durations = [t["end"] - t["start"] for t in agent_turns]
    caller_durations = [t["end"] - t["start"] for t in caller_turns]

    total_agent_time = sum(agent_durations)
    total_caller_time = sum(caller_durations)

    # Analisis de respuesta: cuando el agente termina, cuanto tarda el caller en responder
    response_latencies = []
    interruptions_by_caller = 0
    interruptions_by_agent = 0
    overlap_duration = 0.0

    # Ordenar cronologicamente
    sorted_turns = sorted(turns, key=lambda x: x["start"])

    for i in range(len(sorted_turns) - 1):
        curr = sorted_turns[i]
        nxt = sorted_turns[i + 1]

        # Overlap general
        if curr["end"] > nxt["start"]:
            overlap = min(curr["end"], nxt["end"]) - nxt["start"]
            overlap_duration += max(0.0, overlap)

        # Si el agente hablaba y el caller viene despues o durante
        if curr["channel"] == 1 and nxt["channel"] == 0:
            latency = nxt["start"] - curr["end"]
            response_latencies.append(latency)
            if latency < 0:
                interruptions_by_caller += 1

        # Si el caller hablaba y el agente lo interrumpe
        if curr["channel"] == 0 and nxt["channel"] == 1:
            if nxt["start"] < curr["end"]:
                interruptions_by_agent += 1

    # Trampa de silencios largos del agente (>3.5s entre un turno del agente y el siguiente)
    # Ver si el caller habla durante ese silencio
    long_silence_reactions = 0
    long_silences_count = 0
    for i in range(len(agent_turns) - 1):
        gap = agent_turns[i + 1]["start"] - agent_turns[i]["end"]
        if gap >= 3.5:
            long_silences_count += 1
            # Revisar si el caller hablo dentro de este gap
            spoke = any(
                t["start"] >= agent_turns[i]["end"] and t["start"] < agent_turns[i + 1]["start"]
                for t in caller_turns
            )
            if spoke:
                long_silence_reactions += 1

    pos_latencies = [l for l in response_latencies if l >= 0]

    return {
        "agent_turn_count": len(agent_turns),
        "caller_turn_count": len(caller_turns),
        "total_caller_time": total_caller_time,
        "caller_turn_dur_mean": float(np.mean(caller_durations)) if caller_durations else 0.0,
        "caller_turn_dur_std": float(np.std(caller_durations)) if caller_durations else 0.0,
        "latency_mean": float(np.mean(response_latencies)) if response_latencies else 0.0,
        "latency_std": float(np.std(response_latencies)) if response_latencies else 0.0,
        "pos_latency_mean": float(np.mean(pos_latencies)) if pos_latencies else 0.0,
        "pos_latency_std": float(np.std(pos_latencies)) if pos_latencies else 0.0,
        "overlap_duration": overlap_duration,
        "interruptions_by_caller": interruptions_by_caller,
        "interruptions_by_agent": interruptions_by_agent,
        "long_silences_count": long_silences_count,
        "long_silence_reaction_ratio": (long_silence_reactions / long_silences_count) if long_silences_count > 0 else 0.0,
    }


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    results = []

    for _, row in manifest.iterrows():
        turn_file = TURNS_DIR / f"{row['anon_id']}.json"
        if not turn_file.exists():
            continue

        with open(turn_file, "r") as f:
            data = json.load(f)

        features = analyze_turns(data.get("turns", []))
        features["anon_id"] = row["anon_id"]
        features["label"] = row["label"]
        features["split"] = row["split"]
        features["duration_s"] = row["duration_s"]
        results.append(features)

    df = pd.DataFrame(results)
    print(f"\n=======================================================")
    print(f"ANÁLISIS EXPLORATORIO DE SEÑAL CONVERSACIONAL ({len(df)} llamadas)")
    print(f"=======================================================\n")

    train_df = df[df["split"] == "train"]
    print("MÉTRICAS PROMEDIO EN TRAIN (HUMANO vs SINTÉTICO):")
    cols_to_compare = [
        "latency_mean",
        "pos_latency_mean",
        "pos_latency_std",
        "caller_turn_dur_mean",
        "caller_turn_dur_std",
        "overlap_duration",
        "interruptions_by_caller",
        "interruptions_by_agent",
        "long_silence_reaction_ratio",
        "caller_turn_count",
    ]

    summary = train_df.groupby("label")[cols_to_compare].mean().T
    summary["Diff % (Synth vs Hum)"] = ((summary["synthetic"] - summary["human"]) / summary["human"]) * 100
    print(summary.to_string())

    # Entrenar un clasificador rápido de prueba en Train y evaluar en Val
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in cols_to_compare]
    train_data = df[df["split"] == "train"].dropna(subset=feature_cols)
    val_data = df[df["split"] == "val"].dropna(subset=feature_cols)

    X_train = train_data[feature_cols]
    y_train = (train_data["label"] == "synthetic").astype(int)

    X_val = val_data[feature_cols]
    y_val = (val_data["label"] == "synthetic").astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    val_preds_prob = clf.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_preds_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_preds_prob)

    print(f"\n=======================================================")
    print(f"CLASIFICADOR CONVERSACIONAL BASELINE (Regresión Logística)")
    print(f"Accuracy en Val: {acc:.4f} ({acc*100:.1f}%)")
    print(f"ROC-AUC en Val:  {auc:.4f}")
    print(f"=======================================================\n")

    # Importancia de features según coeficientes de Regresión Logística
    coefs = pd.Series(clf.coef_[0], index=feature_cols).sort_values(ascending=False)
    print("PESOS DE LAS CARACTERÍSTICAS (Positivo = predice Sintético, Negativo = predice Humano):")
    for feat, coef in coefs.items():
        print(f"  {feat:30s}: {coef:+.4f}")


if __name__ == "__main__":
    main()
