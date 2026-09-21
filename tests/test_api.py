from fastapi.testclient import TestClient
import pytest

from app.main import create_app


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"model": "laya-rl-agent", "answers": {"q": {"type": questions["q"]["type"]}}, "usage": {"input_tokens": 3, "output_tokens": 0}}


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "Route this", "criteria": {"billing": "payments", "other": "anything else"}},
        {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "medium", "high"]},
        {"type": "noul", "instructions": "Is a refund requested?"},
    ],
)
def test_predict_returns_native_result(question):
    agent = FakeAgent()
    with TestClient(create_app(lambda: agent)) as client:
        response = client.post("/predict", json={"state": {"body": "Please refund me"}, "questions": {"q": question}})
        assert response.status_code == 200
        assert response.json() == {"model": "laya-rl-agent", "answers": {"q": {"type": question["type"]}}, "usage": {"input_tokens": 3, "output_tokens": 0}}
        assert agent.calls == [({"body": "Please refund me"}, {"q": question})]
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"state": "text", "questions": {}},
        {"state": "text", "questions": {"q": {"type": "unknown", "instructions": "x"}}},
        {"state": "text", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}}},
        {"state": "text", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}},
        {"state": "text", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": []}}},
        {"state": 4, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    ],
)
def test_invalid_input_is_rejected(body):
    agent = FakeAgent()
    with TestClient(create_app(lambda: agent)) as client:
        assert client.post("/predict", json=body).status_code == 422
        assert agent.calls == []


def test_startup_fails_if_model_does_not_load():
    def fail():
        raise FileNotFoundError("checkpoint missing")

    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        with TestClient(create_app(fail)):
            pass
