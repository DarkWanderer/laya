import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app import main as api
from app.main import create_app, load_agent


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"model": "laya-rl-agent", "answers": {name: NATIVE_ANSWERS[q["type"]] for name, q in questions.items()}, "usage": {"input_tokens": 3, "output_tokens": 0}}


# Laya 0.3.5 output, including fields the Decisions response does not carry.
NATIVE_ANSWERS = {
    "choice": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1}, "confidence": 0.8, "action": {"act_probability": 0.7}},
    "score": {"type": "score", "score": 1.5, "legend": {"0": "low", "1": "medium", "2": "high"}, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.5, "action": {"act_probability": 0.7}},
    "noul": {"type": "noul", "noul": 0.96, "confidence": 0.96, "action": {"act_probability": 0.7}},
}

DECISIONS_ANSWERS = {
    "choice": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": {"billing": 0.9, "other": 0.1}},
    "score": {"type": "score", "score": 1.5, "confidence": 0.5, "legend": {"0": "low", "1": "medium", "2": "high"}, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
    "noul": {"type": "noul", "noul": 0.96},
}

URL = "/api/alpha/decisions"
MODEL = "convaiinnovations/laya"


def test_load_agent_uses_cuda(monkeypatch):
    calls = []
    agent = FakeAgent()

    def load(path, device):
        calls.append((path, device))
        return agent

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=load))
    assert load_agent().agent is agent
    assert calls == [("/opt/model", "cuda")]


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "Route this", "criteria": {"billing": "payments", "other": None}},
        {"type": "choice", "instructions": {"task": "Route this"}, "criteria": {"billing": {"covers": ["refunds"]}}},
        {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "medium", "high"]},
        {"type": "score", "instructions": ["Rate urgency"], "criteria": [{"level": "low"}]},
        {"type": "noul", "instructions": "Is a refund requested?"},
        {"type": "noul", "instructions": "Is a refund requested?", "criteria": {"true": "asks for money back", "false": {"other": "anything else"}}},
    ],
)
def test_decisions_returns_decisions_shape(question):
    agent = FakeAgent()
    with TestClient(create_app(lambda: agent)) as client:
        response = client.post(URL, json={"model": MODEL, "state": {"body": "Please refund me"}, "questions": {"q": question}})
        assert response.status_code == 200
        body = response.json()
        assert body.pop("id").startswith("gen-dec-")
        assert body == {"model": MODEL, "answers": {"q": DECISIONS_ANSWERS[question["type"]]}, "usage": {"input_tokens": 3, "output_tokens": 0}}
        assert agent.calls == [({"body": "Please refund me"}, {"q": question})]
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200


def test_decisions_accepts_openrouter_request_fields():
    agent = FakeAgent()
    body = {
        "model": MODEL,
        "state": [{"role": "user", "content": "Please refund me"}],
        "questions": {"q": {"type": "noul", "instructions": "Is a refund requested?"}},
        "provider": {"order": ["TypeSafe"]},
        "session_id": "session-1234",
        "trace": {"trace_id": "t"},
        "user": "user-1",
    }
    with TestClient(create_app(lambda: agent)) as client:
        assert client.post(URL, json=body).status_code == 200
        assert agent.calls == [(body["state"], body["questions"])]


@pytest.mark.parametrize(
    "body",
    [
        {"model": MODEL, "state": "text", "questions": {}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "unknown", "instructions": "x"}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "score", "instructions": "x", "criteria": []}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": []}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "yes"}}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": 4}}},
        {"model": MODEL, "state": 4, "questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": "x"}}, "session_id": "s" * 257},
        {"state": "text", "questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"model": "typesafe/jev-1.13", "state": "text", "questions": {"q": {"type": "noul", "instructions": "x"}}},
    ],
)
def test_invalid_input_is_rejected(body):
    agent = FakeAgent()
    with TestClient(create_app(lambda: agent)) as client:
        response = client.post(URL, json=body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == 400
        assert error["message"]
        assert agent.calls == []


def test_unknown_route_uses_error_shape():
    with TestClient(create_app(FakeAgent)) as client:
        assert client.post("/predict", json={}).json() == {"error": {"code": 404, "message": "Not Found"}}
        wrong_method = client.get(URL)
        assert wrong_method.json() == {"error": {"code": 405, "message": "Method Not Allowed"}}
        assert wrong_method.headers["allow"] == "POST"


def test_startup_fails_if_model_does_not_load():
    def fail():
        raise FileNotFoundError("checkpoint missing")

    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        with TestClient(create_app(fail)):
            pass


def test_long_conversation_is_rejected_before_inference():
    from laya.agent import Agent
    from transformers import AutoTokenizer

    class TokenizedFakeAgent(FakeAgent):
        tok = AutoTokenizer.from_pretrained("/opt/model/tokenizer", local_files_only=True)
        cfg = {"max_len": 512, "head_max_len": 192}
        _to_internal = staticmethod(Agent._to_internal)

    agent = TokenizedFakeAgent()
    question = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"}}
    with TestClient(create_app(lambda: api.LengthCheckedAgent(agent))) as client:
        short = client.post(URL, json={"model": MODEL, "state": "Please refund my duplicate payment.", "questions": {"q": question}})
        assert short.status_code == 200
        agent.calls.clear()

        long = client.post(URL, json={"model": MODEL, "state": "Please refund my duplicate payment. " + "A neutral follow-up. " * 200, "questions": {"q": question}})
        assert long.status_code == 400
        assert "'q'" in long.json()["error"]["message"]
        assert "512" in long.json()["error"]["message"]
        assert agent.calls == []
