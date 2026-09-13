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


_DIGIT_WORD_TO_DIGIT = {
    "CERO": "0", "UNO": "1", "DOS": "2", "TRES": "3", "CUATRO": "4",
    "CINCO": "5", "SEIS": "6", "SIETE": "7", "OCHO": "8", "NUEVE": "9",
}


def liveness_code_matches(digit_words: str, transcript: str) -> bool:
    """Compara el codigo de vivacidad (ej. 'CINCO DOS SIETE CERO') contra la
    transcripcion, aceptando dos formatos validos de la MISMA respuesta
    correcta:
    1. Como palabras sueltas ("cinco dos siete cero") -- usa phrase_matches.
    2. Como numero pegado ("5270") -- lo mas comun en la practica: Whisper
       casi siempre transcribe digitos hablados como numero, no como palabras
       (confirmado en produccion el 13-sep-2026: alguien dijo el codigo bien
       y transcribio "5270", y el chequeo de solo-palabras lo rechazo)."""
    if phrase_matches(digit_words, transcript):
        return True

    expected_digits = "".join(_DIGIT_WORD_TO_DIGIT[w] for w in digit_words.split())
    transcript_digits = "".join(ch for ch in transcript if ch.isdigit())
    return bool(expected_digits) and expected_digits in transcript_digits
