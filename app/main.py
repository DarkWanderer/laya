"""Local HTTP boundary for the English Laya checkpoint."""

import asyncio
from contextlib import asynccontextmanager
from sys import maxsize
from typing import Annotated, Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: StrictStr = Field(min_length=1)
    criteria: dict[StrictStr, StrictStr | None] | list[StrictStr]

    @field_validator("criteria")
    @classmethod
    def at_least_two_options(cls, criteria):
        if len(criteria) < 2:
            raise ValueError("choice needs at least two options")
        return criteria


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["score"]
    instructions: StrictStr = Field(min_length=1)
    criteria: list[StrictStr] = Field(min_length=2)


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    instructions: StrictStr = Field(min_length=1)


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: StrictStr | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)


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

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(request: PredictRequest):
        questions = {name: question.model_dump(exclude_none=True) for name, question in request.questions.items()}
        async with app.state.inference_lock:
            try:
                return await run_in_threadpool(app.state.agent.predict, request.state, questions)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
