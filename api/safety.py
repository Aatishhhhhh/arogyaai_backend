"""Small precautionary safety gate. This is NOT a validated clinical triage tool."""

import re

# Deliberately conservative, high-signal phrases for a first API-layer guard.
# This must be expanded and clinically reviewed before any public medical launch.
_URGENT_PATTERNS = [
    r"\b(can(?:not|'t) breathe|unable to breathe|struggling to breathe)\b",
    r"\b(severe difficulty breathing|severe shortness of breath)\b",
    r"\b(i am having chest pain|i have chest pain right now|crushing chest pain)\b",
    r"\b(face drooping|sudden weakness on one side|slurred speech suddenly)\b",
    r"\b(unconscious|not waking up|severe bleeding|bleeding won't stop)\b",
    r"\b(overdose|took too many pills)\b",
    r"\b(i want to kill myself|i am going to kill myself|i might hurt myself)\b",
    r"\b(saans nahi aa rahi|saans lene mein dikkat|saans lene me dikkat)\b",
    r"\b(behosh ho gaya|behosh ho gayi|bahut khoon beh raha hai)\b",
]

_NEGATION_PREFIX = re.compile(r"\b(no|not|never|don't|dont|doesn't|without)\s+(?:have\s+)?(?:any\s+)?$", re.I)

EMERGENCY_GUIDANCE = (
    "This may be urgent. If these symptoms are happening now, contact local emergency "
    "services or go to the nearest emergency department immediately. Do not wait for an AI "
    "response. If you can, ask someone nearby to stay with you. This is precautionary guidance, "
    "not a diagnosis."
)


def has_urgent_signal(message: str) -> bool:
    """Return True for a small set of high-signal urgent phrases.

    Keyword matching can miss emergencies and can also flag educational questions. Never treat
    a False result as proof that the user is safe.
    """
    text = re.sub(r"\s+", " ", message.casefold().replace("’", "'"))
    for pattern in _URGENT_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        prefix = text[max(0, match.start() - 28):match.start()]
        if _NEGATION_PREFIX.search(prefix):
            continue
        return True
    return False
