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
