from bank.phrase_generator import generate_confirmation_phrase
from bank.phrase_match import phrase_matches


def test_generated_phrase_has_expected_anchor():
    phrase = generate_confirmation_phrase()
    assert phrase.startswith("CONFIRMO LA TRANSFERENCIA CON CLAVE ")


def test_generated_phrases_vary():
    phrases = {generate_confirmation_phrase() for _ in range(20)}
    assert len(phrases) > 1  # no siempre la misma clave


def test_generated_phrase_matches_itself_spoken_naturally():
    phrase = generate_confirmation_phrase()
    spoken = phrase.lower()
    assert phrase_matches(phrase, spoken) is True


def test_generated_phrase_does_not_match_a_different_one():
    a = generate_confirmation_phrase()
    b = generate_confirmation_phrase()
    if a != b:
        assert phrase_matches(a, b.lower()) is False
