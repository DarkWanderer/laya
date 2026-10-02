import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app import main as api
from app.main import create_app, load_agent


class FakeAgent:
    def __init__(self):
        self.calls = []
        self.cfg = {"max_len": 1024, "head_max_len": 256}

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
MODEL = "convaiinnovations/laya-multilingual"


def test_load_agent_uses_cuda(monkeypatch):
    calls = []
    agent = FakeAgent()

    def load(path, device):
        calls.append((path, device))
        return agent

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=load))
    assert load_agent().agent is agent
    assert calls == [("/opt/model", "cuda")]
    assert agent.cfg == {"max_len": 4096, "head_max_len": 256}


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
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": ""}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "choice", "instructions": {}, "criteria": {"a": None}}}},
        {"model": MODEL, "state": "text", "questions": {"q": {"type": "score", "instructions": [], "criteria": ["a"]}}},
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


def test_openapi_documents_actual_error_shape():
    with TestClient(create_app(FakeAgent)) as client:
        schema = client.get("/openapi.json").json()
    responses = schema["paths"][URL]["post"]["responses"]
    assert "422" not in responses
    assert responses["200"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/DecisionsResponse"}
    for code in ("400", "500"):
        assert responses[code]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    # Laya echoes each score criterion into the legend, so structured criteria appear there too.
    legend = schema["components"]["schemas"]["ScoreAnswer"]["properties"]["legend"]["additionalProperties"]
    assert {"type": "object"} in [{"type": option.get("type")} for option in legend["anyOf"]]


def test_unknown_route_uses_error_shape():
    with TestClient(create_app(FakeAgent)) as client:
        assert client.post("/predict", json={}).json() == {"error": {"code": 404, "message": "Not Found"}}
        wrong_method = client.get(URL)
        assert wrong_method.json() == {"error": {"code": 405, "message": "Method Not Allowed"}}
        assert wrong_method.headers["allow"] == "POST"


def test_structured_http_error_detail_is_json():
    app = create_app(FakeAgent)

    @app.get("/structured")
    async def structured():
        raise api.HTTPException(status_code=409, detail={"reason": "busy"})

    with TestClient(app) as client:
        assert client.get("/structured").json() == {"error": {"code": 409, "message": '{"reason": "busy"}'}}


class FailingAgent:
    def predict(self, state, questions):
        raise KeyError("boom")


def test_unhandled_error_still_propagates_for_logging():
    body = {"model": MODEL, "state": "text", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    # Starlette re-raises after the handler responds, so uvicorn still logs the traceback.
    with pytest.raises(KeyError, match="boom"):
        with TestClient(create_app(FailingAgent)) as client:
            client.post(URL, json=body)
    with TestClient(create_app(FailingAgent), raise_server_exceptions=False) as client:
        assert client.post(URL, json=body).json() == {"error": {"code": 500, "message": "Internal Server Error"}}


def test_startup_fails_if_model_does_not_load():
    def fail():
        raise FileNotFoundError("checkpoint missing")

    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        with TestClient(create_app(fail)):
            pass


def test_long_conversation_is_rejected_before_inference():
    from laya.agent import Agent
    from laya.common import build_sequence
    from transformers import AutoTokenizer

    class TokenizedFakeAgent(FakeAgent):
        tok = AutoTokenizer.from_pretrained("/opt/model/tokenizer", local_files_only=True)
        _to_internal = staticmethod(Agent._to_internal)

    agent = TokenizedFakeAgent()
    agent.cfg["max_len"] = api.MAX_LEN
    question = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"}}

    def state(repeats):
        return "Please refund my duplicate payment. " + "A neutral follow-up. " * repeats

    def tokens(repeats):
        return len(build_sequence(agent.tok, state(repeats), Agent._to_internal(question), max_len=sys.maxsize, head_max_len=agent.cfg["head_max_len"])[0])

    # Guard the fixtures: the accepted state must not fit the checkpoint's original window, and the rejected one must not fit the expanded one.
    assert 1024 < tokens(400) <= api.MAX_LEN < tokens(1000)
    with TestClient(create_app(lambda: api.LengthCheckedAgent(agent))) as client:
        within = client.post(URL, json={"model": MODEL, "state": state(400), "questions": {"q": question}})
        assert within.status_code == 200
        agent.calls.clear()

        long = client.post(URL, json={"model": MODEL, "state": state(1000), "questions": {"q": question}})
        assert long.status_code == 400
        assert "'q'" in long.json()["error"]["message"]
        assert str(api.MAX_LEN) in long.json()["error"]["message"]
        assert agent.calls == []
