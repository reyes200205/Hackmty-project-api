# Guion de "trampas" que replica la dinamica del dataset del reto:
# confirmar datos, preguntar por algo inexistente, e interrumpir/callar a proposito.

CALL_SCRIPT = [
    {
        "type": "speak",
        # {name_part} se rellena en runner.py: ", María" si se reconocio al cliente
        # por su numero, o "" si es un numero desconocido.
        "text": "Hola{name_part}, buenas tardes, le habla el asistente virtual del banco. "
                "Para poder ayudarle, ¿me podría confirmar su nombre completo, por favor?",
    },
    {"type": "listen", "seconds": 2.5},  # se corta antes de que termine: trampa de interrupcion
    {
        "type": "speak",
        "text": "Perfecto, gracias. Ahora, ¿me puede confirmar los últimos cuatro dígitos de su tarjeta?",
    },
    {"type": "listen", "seconds": 4.0},
    {
        "type": "speak",
        "text": "Veo que tiene activado nuestro Seguro de Vida Flexible Plus. "
                "¿Desea hacer alguna aclaración sobre ese producto?",
    },
    {"type": "listen", "seconds": 4.0},
    {"type": "silence", "seconds": 6.0},  # trampa de silencio prolongado
    {
        "type": "speak",
        "text": "Disculpe la espera, ya verifiqué su información correctamente. "
                "Gracias por llamar, que tenga buen día.",
    },
]
