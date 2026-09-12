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
    (mayusculas, acentos, puntuacion, relleno como 'eh') pero exige que CADA
    palabra de la frase esperada aparezca (exacta o casi) en lo dicho.

    Importante: se compara palabra por palabra, no la cadena completa. Con
    frases dinamicas que comparten un prefijo largo (ej. "CONFIRMO LA
    TRANSFERENCIA CON CLAVE X Y"), un ratio de cadena completa deja que el
    prefijo compense una palabra clave incorrecta (ratio ~0.93 aunque la
    clave real sea otra) — palabra por palabra evita ese hueco."""
    if not spoken:
        return False

    expected_words = normalize(expected).split()
    spoken_words = normalize(spoken).split()
    if not expected_words:
        return False

    for word in expected_words:
        if word in spoken_words:
            continue
        best = max((difflib.SequenceMatcher(None, word, sw).ratio() for sw in spoken_words), default=0.0)
        if best < threshold:
            return False
    return True
