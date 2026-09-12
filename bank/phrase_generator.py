import secrets

# Palabras elegidas por ser faciles de pronunciar y distintas entre si (para
# que el reconocimiento de voz de Twilio las confunda lo menos posible).
_WORDS_A = ["TIGRE", "AGUILA", "DELFIN", "LEON", "PANTERA", "HALCON", "JAGUAR", "LOBO"]
_WORDS_B = ["MONTANA", "OCEANO", "DESIERTO", "VOLCAN", "GLACIAR", "BOSQUE", "RIO", "VALLE"]


def generate_confirmation_phrase() -> str:
    """Frase de confirmacion unica por transferencia: ancla fija (ayuda al
    reconocimiento de voz a ubicar el contexto) + clave dinamica de 2 palabras
    (evita que sirva una respuesta generica o grabada de antemano)."""
    w1 = secrets.choice(_WORDS_A)
    w2 = secrets.choice(_WORDS_B)
    return f"CONFIRMO LA TRANSFERENCIA CON CLAVE {w1} {w2}"
