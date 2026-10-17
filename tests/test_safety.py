from api.safety import has_urgent_signal


def test_detects_obvious_urgent_phrase():
    assert has_urgent_signal("I can't breathe") is True


def test_detects_hinglish_urgent_phrase():
    assert has_urgent_signal("Mujhe saans nahi aa rahi") is True


def test_does_not_flag_unrelated_health_question():
    assert has_urgent_signal("What causes dry skin in winter?") is False


def test_does_not_treat_negated_chest_pain_as_current_emergency():
    assert has_urgent_signal("I do not have chest pain right now") is False
