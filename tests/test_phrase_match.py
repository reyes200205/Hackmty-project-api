from bank.phrase_match import phrase_matches

EXPECTED = "CONFIRMO LA TRANSFERENCIA"


def test_exact_lowercase_matches():
    assert phrase_matches(EXPECTED, "confirmo la transferencia") is True


def test_wrong_phrase_does_not_match():
    assert phrase_matches(EXPECTED, "cancelo la transferencia") is False


def test_empty_speech_does_not_match():
    assert phrase_matches(EXPECTED, "") is False


def test_minor_recognition_noise_still_matches():
    assert phrase_matches(EXPECTED, "eh, confirmo la transferencia") is True


def test_accents_and_case_are_ignored():
    assert phrase_matches("CONFÍRMO LA TRANSFERÉNCIA", "confirmo la transferencia") is True


def test_unrelated_phrase_does_not_match():
    assert phrase_matches(EXPECTED, "buenas tardes como esta") is False


def test_dynamic_phrase_wrong_second_keyword_does_not_match():
    """Regresion: con frases dinamicas de prefijo largo, un ratio de cadena
    completa dejaba que el prefijo compensara una clave incorrecta."""
    expected = "CONFIRMO LA TRANSFERENCIA CON CLAVE TIGRE MONTANA"
    assert phrase_matches(expected, "confirmo la transferencia con clave tigre oceano") is False


def test_dynamic_phrase_wrong_first_keyword_does_not_match():
    expected = "CONFIRMO LA TRANSFERENCIA CON CLAVE TIGRE MONTANA"
    assert phrase_matches(expected, "confirmo la transferencia con clave leon montana") is False


def test_dynamic_phrase_correct_keywords_match():
    expected = "CONFIRMO LA TRANSFERENCIA CON CLAVE TIGRE MONTANA"
    assert phrase_matches(expected, "confirmo la transferencia con clave tigre montana") is True
