import secrets

# Palabras elegidas por ser faciles de pronunciar y distintas entre si (para
# que el reconocimiento de voz de Twilio las confunda lo menos posible).
_WORDS_A = ["TIGRE", "AGUILA", "DELFIN", "LEON", "PANTERA", "HALCON", "JAGUAR", "LOBO"]
_WORDS_B = ["MONTANA", "OCEANO", "DESIERTO", "VOLCAN", "GLACIAR", "BOSQUE", "RIO", "VALLE"]

# Digitos para el codigo de vivacidad (ver generate_liveness_word). No es un
# secreto que haya que proteger -- su valor esta en que no existe todavia
# cuando alguien prepara un audio con anticipacion (ver nota en
# confirmation_service.py).
#
# FIX (13-sep-2026): antes esto era 1 palabra de una lista fija de 8
# (ROJO/AZUL/VERDE/...). Confirmado en produccion que eso no sirve: un
# atacante preparado puede generar las 8 palabras posibles CON ANTICIPACION
# (toma un par de minutos), y cuando el agente anuncia cual le toco, solo
# reproduce el audio correcto -- eso explica por que Google Translate
# respondio en 0.49s, no estaba generando nada en vivo. Con 4 digitos al azar
# (10,000 combinaciones) pre-generar todas las posibles ya no es practico.
_DIGIT_WORDS = ["CERO", "UNO", "DOS", "TRES", "CUATRO", "CINCO", "SEIS", "SIETE", "OCHO", "NUEVE"]


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
    """Codigo de vivacidad: 4 digitos al azar (dichos como palabras, para
    reusar el mismo comparador difuso de phrase_matches), generados al momento
    de iniciar la llamada -- nunca antes y nunca en la app. No protege un
    secreto -- protege que la respuesta se produzca en vivo: con 10,000
    combinaciones posibles, pre-generar todas de antemano no es practico."""
    return " ".join(secrets.choice(_DIGIT_WORDS) for _ in range(4))
