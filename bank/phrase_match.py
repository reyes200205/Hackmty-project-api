import difflib
import re
import unicodedata

MATCH_THRESHOLD = 0.85


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_matches(expected: str, spoken: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """Compara tolerando diferencias normales de reconocimiento de voz
    (mayusculas, acentos, puntuacion) pero sin aceptar cualquier respuesta."""
    if not spoken:
        return False
    a, b = normalize(expected), normalize(spoken)
    if a == b:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold
