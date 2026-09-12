import re


def normalize_phone(raw: str) -> str:
    """Normaliza a formato E.164 basico: solo digitos con un '+' al inicio.
    Twilio ya manda el campo 'From' en E.164, esto es principalmente para
    normalizar los numeros que se capturan a mano al sembrar la base."""
    digits = re.sub(r"[^\d]", "", raw)
    if raw.strip().startswith("+"):
        return "+" + digits
    if len(digits) == 10:  # numero de 10 digitos sin lada de pais (asume +1)
        return "+1" + digits
    return "+" + digits
