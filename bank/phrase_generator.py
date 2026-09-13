import secrets

# Palabras elegidas por ser faciles de pronunciar y distintas entre si (para
# que el reconocimiento de voz de Twilio las confunda lo menos posible).
_WORDS_A = ["TIGRE", "AGUILA", "DELFIN", "LEON", "PANTERA", "HALCON", "JAGUAR", "LOBO"]
_WORDS_B = ["MONTANA", "OCEANO", "DESIERTO", "VOLCAN", "GLACIAR", "BOSQUE", "RIO", "VALLE"]

# Palabras distintas a las de arriba, a proposito: esta palabra NUNCA se
# muestra en la app, solo se genera al momento de la llamada y la dice el
# agente en voz alta (ver generate_liveness_word). No es un secreto que haya
# que proteger -- su valor esta en que no existe todavia cuando alguien
# prepara un audio con anticipacion (ver nota en confirmation_service.py).
_LIVENESS_WORDS = ["ROJO", "AZUL", "VERDE", "AMARILLO", "MORADO", "NARANJA", "GRIS", "BLANCO"]


def generate_confirmation_phrase() -> str:
    """Frase de confirmacion unica por transferencia: ancla fija (ayuda al
    reconocimiento de voz a ubicar el contexto) + clave dinamica de 2 palabras
    (evita que sirva una respuesta generica o grabada de antemano).

    Se genera y se muestra en la app en cuanto el cliente pide la transferencia
    -- es decir, con minutos de anticipacion a la llamada de confirmacion. Eso
    es suficiente tiempo para que alguien con acceso a la app prepare un audio
    sintetico con esta frase antes de que suene el telefono (ver
    generate_liveness_word, que cierra ese hueco)."""
    w1 = secrets.choice(_WORDS_A)
    w2 = secrets.choice(_WORDS_B)
    return f"CONFIRMO LA TRANSFERENCIA CON CLAVE {w1} {w2}"


def generate_liveness_word() -> str:
    """Palabra de vivacidad: se genera al momento de iniciar la llamada
    (segundos antes de que el agente la diga), nunca antes y nunca en la app.
    No protege un secreto -- protege que la respuesta se produzca en vivo: es
    imposible tener listo un audio con una palabra que no existia cuando se
    preparo el ataque."""
    return secrets.choice(_LIVENESS_WORDS)
