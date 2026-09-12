import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

from detector.conversational import (
    extract_conversational_features,
    extract_features_from_turns,
    predict_conversational,
)

HACKMTY_DIR = Path("../hackmty26")
AUDIO_DIR = HACKMTY_DIR / "audio"
TURNS_DIR = HACKMTY_DIR / "turns"
MANIFEST_PATH = HACKMTY_DIR / "manifest.csv"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    val_sample = manifest[manifest["split"] == "val"].head(5)

    print("\n=======================================================")
    print("TEST DE EXTRACCIÓN CONVERSACIONAL SOBRE AUDIO REAL (.wav)")
    print("=======================================================\n")

    for _, row in val_sample.iterrows():
        call_id = row["anon_id"]
        label = row["label"]
        wav_path = AUDIO_DIR / f"{call_id}.wav"
        turn_path = TURNS_DIR / f"{call_id}.json"

        if not wav_path.exists():
            continue

        # Cargar audio estereo
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        caller = data[:, 0]
        agent = data[:, 1]
        duration_s = len(caller) / sr

        # Medir tiempo de inferencia VAD + features conversacionales
        t0 = time.perf_counter()
        is_synth, conf, feats = predict_conversational(caller, agent, sr)
        inference_time_ms = (time.perf_counter() - t0) * 1000

        # Cargar ground truth de turns
        with open(turn_path, "r") as f:
            gt_turns = json.load(f).get("turns", [])
        gt_feats = extract_features_from_turns(gt_turns)

        print(f"Llamada: {call_id} ({duration_s:.1f}s) | Real: {label.upper()}")
        print(f"  -> Predicción Conversacional: {'SINTÉTICO' if is_synth else 'HUMANO'} (conf: {conf:.3f})")
        print(f"  -> Tiempo de cómputo DSP: {inference_time_ms:.2f} ms")
        print(f"  -> VAD extraído: pos_latency_mean = {feats['pos_latency_mean']:.2f}s | GT = {gt_feats['pos_latency_mean']:.2f}s")
        print(f"  -> VAD extraído: overlap_duration = {feats['overlap_duration']:.2f}s | GT = {gt_feats['overlap_duration']:.2f}s\n")


if __name__ == "__main__":
    main()
