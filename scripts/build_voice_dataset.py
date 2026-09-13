"""Extrae segmentos cortos del canal del caller (usando los turns ya provistos
por el dataset del reto) y calcula features acusticas crudas (detector.features)
para entrenar un clasificador de voz sintetica calibrado para CLIPS CORTOS
(como los de la confirmacion de transferencias), no para llamadas completas.

Uso: python scripts/build_voice_dataset.py
Genera: scripts/voice_dataset.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.features import FEATURE_NAMES, extract_features  # noqa: E402

HACKMTY26_DIR = PROJECT_ROOT.parent / "hackmty26"
MANIFEST_PATH = HACKMTY26_DIR / "manifest.csv"
AUDIO_DIR = HACKMTY26_DIR / "audio"
TURNS_DIR = HACKMTY26_DIR / "turns"

MIN_SEG_SECONDS = 1.5
MAX_SEG_SECONDS = 15.0
OUT_CSV = Path(__file__).parent / "voice_dataset.csv"


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_manifest()
    print(f"{len(rows)} llamadas en el manifest")

    out_rows = []
    skipped_calls = 0

    for i, row in enumerate(rows):
        anon_id = row["anon_id"]
        label = row["label"]  # human | synthetic
        split = row["split"]  # train | val

        wav_path = AUDIO_DIR / f"{anon_id}.wav"
        turns_path = TURNS_DIR / f"{anon_id}.json"
        if not wav_path.exists() or not turns_path.exists():
            skipped_calls += 1
            continue

        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        caller = data[:, 0]

        with open(turns_path, encoding="utf-8") as f:
            turns = json.load(f)["turns"]

        caller_turns = [t for t in turns if t["channel"] == 0]

        for t in caller_turns:
            dur = t["end"] - t["start"]
            if dur < MIN_SEG_SECONDS or dur > MAX_SEG_SECONDS:
                continue
            start_idx = int(t["start"] * sample_rate)
            end_idx = int(t["end"] * sample_rate)
            segment = caller[start_idx:end_idx]
            if len(segment) < sample_rate * MIN_SEG_SECONDS:
                continue

            feats = extract_features(segment, sample_rate)
            out_rows.append({
                "anon_id": anon_id,
                "label": label,
                "split": split,
                "duration": round(dur, 2),
                **{name: feats[name] for name in FEATURE_NAMES},
            })

        if (i + 1) % 50 == 0:
            print(f"  procesadas {i + 1}/{len(rows)} llamadas, {len(out_rows)} segmentos hasta ahora")

    print(f"\n{skipped_calls} llamadas saltadas (audio/turns no encontrados)")
    print(f"{len(out_rows)} segmentos cortos extraidos en total")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["anon_id", "label", "split", "duration"] + FEATURE_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Guardado en {OUT_CSV}")


if __name__ == "__main__":
    main()
