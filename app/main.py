"""Local HTTP boundary for the English Laya checkpoint, shaped like OpenRouter's Decisions API."""

import asyncio
import json
from contextlib import asynccontextmanager
from sys import maxsize
from typing import Annotated, Any, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException


MODEL_ID = "convaiinnovations/laya"

# A plain string, or a JSON object or array of structured guidance.
Guidance = StrictStr | dict[str, Any] | list[Any]
# Instructions are the question itself, so an empty one is a client error.
Instructions = Annotated[StrictStr, Field(min_length=1)] | Annotated[dict[str, Any], Field(min_length=1)] | Annotated[list[Any], Field(min_length=1)]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[StrictStr, Guidance | None] = Field(min_length=1)


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["score"]
    instructions: Instructions
    criteria: list[Guidance] = Field(min_length=1)


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true: Guidance
    false: Guidance


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    instructions: Instructions
    criteria: NoulCriteria | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class DecisionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr
    state: Guidance
    questions: dict[str, Question] = Field(min_length=1)
    # Accepted for client compatibility; routing and observability have no local meaning.
    provider: dict[str, Any] | None = None
    session_id: StrictStr | None = Field(default=None, max_length=256)
    trace: dict[str, Any] | None = None
    user: StrictStr | None = Field(default=None, max_length=256)


# Fields each answer type carries in the Decisions response; Laya's extensions are dropped.
ANSWER_FIELDS = {
    "choice": ("type", "choice", "confidence", "probabilities"),
    "score": ("type", "score", "confidence", "legend", "probabilities"),
    "noul": ("type", "noul"),
}


def to_decisions_response(result: dict[str, Any]) -> dict[str, Any]:
    answers = {
        name: {key: answer[key] for key in ANSWER_FIELDS[answer["type"]] if key in answer}
        for name, answer in result["answers"].items()
    }
    usage = result["usage"]
    return {
        "id": f"gen-dec-{uuid4().hex}",
        "model": MODEL_ID,
        "answers": answers,
        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
    }


def error_response(code: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"code": code, "message": message}}, headers=headers)


def describe_validation_error(exc: RequestValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"] if part != "body")
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts) or "Invalid request parameters"


class LengthCheckedAgent:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, state, questions):
        from laya.common import build_sequence

        max_len = self.agent.cfg.get("max_len", 512)
        head_max_len = self.agent.cfg.get("head_max_len", 192)
        for name, question in questions.items():
            # Use Laya's own formatter without its final slice so overflow is visible.
            sequence, _ = build_sequence(
                self.agent.tok,
                state,
                self.agent._to_internal(question),
                max_len=maxsize,
                head_max_len=head_max_len,
            )
            if len(sequence) > max_len:
                raise ValueError(f"question {name!r} needs {len(sequence)} tokens, exceeding the model limit of {max_len}")
        return self.agent.predict(state, questions)


def load_agent():
    import laya

    return LengthCheckedAgent(laya.load("/opt/model", device="cuda"))


def create_app(loader: Callable[[], Any] = load_agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup must fail if the bundled checkpoint cannot be loaded.
        app.state.agent = loader()
        app.state.inference_lock = asyncio.Lock()
        yield

    app = FastAPI(title="Laya Local API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(400, describe_validation_error(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        return error_response(exc.status_code, message, exc.headers)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        return error_response(500, "Internal Server Error")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/alpha/decisions")
    async def decisions(request: DecisionsRequest):
        if request.model != MODEL_ID:
            raise HTTPException(status_code=400, detail=f"{request.model} is not a valid model ID; this server serves {MODEL_ID}")
        questions = {name: question.model_dump(exclude_none=True) for name, question in request.questions.items()}
        async with app.state.inference_lock:
            try:
                result = await run_in_threadpool(app.state.agent.predict, request.state, questions)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return to_decisions_response(result)

    return app


app = create_app()
