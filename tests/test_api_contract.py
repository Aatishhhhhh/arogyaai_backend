from fastapi.testclient import TestClient

from api import main, providers


def test_health_and_readiness_endpoints(monkeypatch):
    monkeypatch.setattr(main.settings, "ai_provider", "legacy")
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["provider"] == "legacy"


def test_legacy_chat_contract_and_request_id(monkeypatch):
    monkeypatch.setattr(main.settings, "ai_provider", "legacy")

    def fake_pipeline(message, state):
        return {
            "stage": "questions",
            "confidence_score": 0.25,
            "questions": ["How long have you had this symptom?"],
            "clarification_data": None,
            "final_output": None,
        }, state

    monkeypatch.setattr(providers, "pipeline", fake_pipeline)
    client = TestClient(main.app)
    response = client.post(
        "/chat",
        headers={"X-Request-ID": "test-request-123"},
        json={"message": "I have a headache"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == "questions"
    assert payload["provider"] == "legacy"
    assert payload["request_id"] == "test-request-123"
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_urgent_signal_uses_safety_gate(monkeypatch):
    monkeypatch.setattr(main.settings, "ai_provider", "legacy")
    client = TestClient(main.app)
    response = client.post("/v1/chat", json={"message": "I can't breathe"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "safety_gate"
    assert payload["stage"] == "urgent_guidance"
    assert "emergency" in payload["final_output"]["message"].lower()


def test_invalid_payload_does_not_echo_input(monkeypatch):
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
    assert "input" not in response.text.lower()
