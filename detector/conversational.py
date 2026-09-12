import numpy as np
from pathlib import Path
from scipy.ndimage import binary_closing, binary_dilation


def compute_vad_intervals(
    signal: np.ndarray,
    sample_rate: int = 8000,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_speech_ms: int = 150,
    min_silence_ms: int = 300,
) -> list[tuple[float, float]]:
    """Detecta intervalos de voz (inicio, fin) en segundos usando energia RMS adaptativa."""
    if len(signal) == 0:
        return []

    frame_len = int(sample_rate * (frame_ms / 1000.0))
    hop_len = int(sample_rate * (hop_ms / 1000.0))

    if len(signal) < frame_len:
        return []

    # Calcular RMS en ventanas con stride eficiente
    n_frames = (len(signal) - frame_len) // hop_len + 1
    shape = (n_frames, frame_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    # Umbral adaptativo basado en el piso de ruido
    noise_floor = np.percentile(rms, 20)
    peak_energy = np.percentile(rms, 95)
    threshold = max(0.005, noise_floor + 0.15 * (peak_energy - noise_floor))

    is_speech = rms > threshold

    # Hangover: rellenar silencios breves entre palabras de una misma frase
    close_frames = int(min_silence_ms / hop_ms)
    if close_frames > 0:
        is_speech = binary_closing(is_speech, structure=np.ones(close_frames))

    # Agrupar regiones continuas de habla
    min_speech_frames = int(min_speech_ms / hop_ms)
    intervals = []
    in_speech = False
    start_idx = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_idx = i
        elif not active and in_speech:
            in_speech = False
            if (i - start_idx) >= min_speech_frames:
                intervals.append((start_idx * hop_ms / 1000.0, i * hop_ms / 1000.0))

    if in_speech and (len(is_speech) - start_idx) >= min_speech_frames:
        intervals.append((start_idx * hop_ms / 1000.0, len(is_speech) * hop_ms / 1000.0))

    return intervals


def extract_features_from_turns(turns: list[dict]) -> dict[str, float]:
    """Extrae las metricas conversacionales discriminantes."""
    agent_turns = [t for t in turns if t["channel"] == 1]
    caller_turns = [t for t in turns if t["channel"] == 0]

    agent_durations = [t["end"] - t["start"] for t in agent_turns]
    caller_durations = [t["end"] - t["start"] for t in caller_turns]

    response_latencies = []
    interruptions_by_caller = 0
    interruptions_by_agent = 0
    overlap_duration = 0.0

    sorted_turns = sorted(turns, key=lambda x: x["start"])

    for i in range(len(sorted_turns) - 1):
        curr = sorted_turns[i]
        nxt = sorted_turns[i + 1]

        # Solapamiento de voces
        if curr["end"] > nxt["start"]:
            overlap = min(curr["end"], nxt["end"]) - nxt["start"]
            overlap_duration += max(0.0, overlap)

        # Latencia: tiempo que tarda el caller en responder tras hablar el agente
        if curr["channel"] == 1 and nxt["channel"] == 0:
            lat = nxt["start"] - curr["end"]
            response_latencies.append(lat)
            if lat < 0:
                interruptions_by_caller += 1

        # Interrupciones: el agente entra mientras el caller habla
        if curr["channel"] == 0 and nxt["channel"] == 1:
            if nxt["start"] < curr["end"]:
                interruptions_by_agent += 1

    pos_latencies = [l for l in response_latencies if l >= 0]

    # Reaccion a silencios largos del agente (>3.5s)
    long_silences_count = 0
    long_silence_reactions = 0
    for i in range(len(agent_turns) - 1):
        gap = agent_turns[i + 1]["start"] - agent_turns[i]["end"]
        if gap >= 3.5:
            long_silences_count += 1
            spoke = any(
                t["start"] >= agent_turns[i]["end"] and t["start"] < agent_turns[i + 1]["start"]
                for t in caller_turns
            )
            if spoke:
                long_silence_reactions += 1

    return {
        "pos_latency_mean": float(np.mean(pos_latencies)) if pos_latencies else 2.0,
        "pos_latency_std": float(np.std(pos_latencies)) if pos_latencies else 0.5,
        "latency_mean": float(np.mean(response_latencies)) if response_latencies else 1.0,
        "caller_turn_dur_mean": float(np.mean(caller_durations)) if caller_durations else 2.5,
        "caller_turn_dur_std": float(np.std(caller_durations)) if caller_durations else 2.0,
        "overlap_duration": float(overlap_duration),
        "interruptions_by_caller": float(interruptions_by_caller),
        "interruptions_by_agent": float(interruptions_by_agent),
        "long_silence_reaction_ratio": float(long_silence_reactions / long_silences_count) if long_silences_count > 0 else 0.5,
        "caller_turn_count": float(len(caller_turns)),
    }


def extract_conversational_features(
    caller: np.ndarray,
    agent: np.ndarray,
    sample_rate: int = 8000,
) -> dict[str, float]:
    """Procesa el audio crudo de caller y agent en milisegundos y calcula sus features conversacionales."""
    caller_intervals = compute_vad_intervals(caller, sample_rate)
    agent_intervals = compute_vad_intervals(agent, sample_rate)

    turns = []
    for s, e in caller_intervals:
        turns.append({"channel": 0, "start": s, "end": e})
    for s, e in agent_intervals:
        turns.append({"channel": 1, "start": s, "end": e})

    return extract_features_from_turns(turns)


_MODEL_CACHE = None
_MODEL_PATH = Path(__file__).parent / "conversational_model.joblib"


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None and _MODEL_PATH.exists():
        import joblib
        _MODEL_CACHE = joblib.load(_MODEL_PATH)
    return _MODEL_CACHE


def predict_conversational(
    caller: np.ndarray,
    agent: np.ndarray,
    sample_rate: int = 8000,
) -> tuple[bool, float, dict[str, float]]:
    """Calcula si la llamada es sintética analizando la señal conversacional.
    Devuelve (is_synthetic, confidence, features_dict)."""
    features = extract_conversational_features(caller, agent, sample_rate)
    model = _get_model()

    if model is None:
        # Heurística de respaldo si no hay modelo entrenado
        is_synth = features["pos_latency_mean"] > 2.0
        conf = 0.85 if is_synth else 0.85
        return is_synth, conf, features

    scaler = model["scaler"]
    clf = model["classifier"]
    keys = model["feature_keys"]

    x = np.array([[features.get(k, 0.0) for k in keys]])
    x_scaled = scaler.transform(x)
    prob_synth = float(clf.predict_proba(x_scaled)[0, 1])

    is_synthetic = prob_synth >= 0.5
    # La confianza representa qué tan seguro está del veredicto
    confidence = float(np.clip(prob_synth if is_synthetic else (1.0 - prob_synth), 0.5, 0.99))
    return is_synthetic, confidence, features

