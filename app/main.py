"""Local HTTP boundary for the English Laya checkpoint."""

import asyncio
from contextlib import asynccontextmanager
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


def load_agent():
    import laya

    return laya.load("/opt/model", device="cpu")


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
