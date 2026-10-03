"""Local HTTP boundary for the multilingual Laya checkpoint, shaped like OpenRouter's Decisions API."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from sys import maxsize
from typing import Annotated, Any, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException


# Reuse uvicorn's logger so detection results show up in the server output without extra logging config.
logger = logging.getLogger("uvicorn.error")

MODEL_ID = "convaiinnovations/laya-multilingual"
# The checkpoint ships a 1024-token window; its mmBERT encoder supports up to 8192, so longer conversations fit.
MAX_LEN = 4096

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


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Guidance]
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class DecisionsResponse(BaseModel):
    id: str
    model: str
    answers: dict[str, Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]]
    usage: Usage


class ErrorBody(BaseModel):
    code: int
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


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


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        logger.info("GPU detected: %s; running inference on cuda", torch.cuda.get_device_name(0))
        return "cuda"
    logger.info("No GPU detected; falling back to CPU inference")
    return "cpu"


def load_agent():
    import laya

    agent = laya.load("/opt/model", device=select_device())
    # Laya reads the window from cfg on every predict, so the override applies to inference and the length check alike.
    agent.cfg["max_len"] = MAX_LEN
    return LengthCheckedAgent(agent)


def create_app(loader: Callable[[], Any] = load_agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup must fail if the bundled checkpoint cannot be loaded.
        app.state.agent = loader()
        app.state.inference_lock = asyncio.Lock()
        yield

    app = FastAPI(title="Laya Local API", lifespan=lifespan)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            # Validation failures are remapped to the 400 error shape below, so FastAPI's default 422 would misdocument them.
            for path in schema["paths"].values():
                for operation in path.values():
                    operation["responses"].pop("422", None)
            for name in ("HTTPValidationError", "ValidationError"):
                schema["components"]["schemas"].pop(name, None)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi

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

    @app.get("/healthz", responses={500: {"model": ErrorResponse, "description": "Internal Server Error"}})
    async def healthz():
        return {"status": "ok"}

    @app.post(
        "/api/alpha/decisions",
        responses={
            200: {"model": DecisionsResponse},
            400: {"model": ErrorResponse, "description": "Invalid request, unknown model, or input over the token limit"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        },
    )
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
