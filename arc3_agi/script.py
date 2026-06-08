import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Sequence, TypeAlias, cast
from urllib.parse import urlparse

import arc_agi
from arcengine import GameAction, GameState
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn


FrameGrid: TypeAlias = list[list[int]]
FrameStack: TypeAlias = list[FrameGrid]
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def write_frame_to_json_file(frame: Sequence[Any], output_path: str | Path) -> None:
    """Write a FrameDataRaw.frame value to a JSON file."""
    serializable_frame = serialize_frame(frame)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(serializable_frame, file_handle, indent=2)

def serialize_frame(frame: Sequence[Any]) -> FrameStack:
    """Convert frame layers to a strict JSON-serializable integer frame stack."""
    raw_layers = [layer.tolist() if hasattr(layer, "tolist") else layer for layer in frame]

    normalized_layers: FrameStack = []
    for layer in raw_layers:
        if not isinstance(layer, list):
            raise ValueError("Frame layer must be a 2D list-like grid")

        grid: FrameGrid = []
        for row in cast(list[Any], layer):
            if not isinstance(row, list):
                raise ValueError("Frame row must be list-like")
            grid.append([int(value) for value in cast(list[Any], row)])

        normalized_layers.append(grid)

    return normalized_layers

def wait_for_action(action_name: str) -> GameAction:
    """Parse a posted action string into a GameAction enum value."""
    try:
        return GameAction[action_name]
    except KeyError as error:
        valid_actions = ", ".join(action.name for action in GameAction)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action_name}'. Expected one of: {valid_actions}",
        ) from error


class ActionRequest(BaseModel):
    action: str


class ActionResponse(BaseModel):
    state: str
    frame: FrameStack
    step: int
    reset_performed: bool = False


def resolve_api_bind_address() -> tuple[str, int]:
    """Resolve host and port for uvicorn from ARC3_API_URL."""
    base_url = os.getenv("ARC3_API_URL", DEFAULT_API_BASE_URL).strip()
    if not base_url:
        base_url = DEFAULT_API_BASE_URL

    if "://" not in base_url:
        base_url = f"http://{base_url}"

    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        raise ValueError(
            "ARC3_API_URL must include a valid host, for example http://127.0.0.1:8000"
        )

    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 8000

    return host, port


def respond(state: GameState, frame: Sequence[Any], step: int, reset_performed: bool = False) -> ActionResponse:
    """Build the JSON payload returned to the caller after a posted action."""
    return ActionResponse(
        state=state.name,
        frame=serialize_frame(frame),
        step=step,
        reset_performed=reset_performed,
    )


class GameRuntime:
    """Owns a single shared ARC environment used by all requests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._arc = arc_agi.Arcade()
        env = self._arc.make("ls20")
        if env is None:
            raise RuntimeError("Failed to create environment")
        observation_space = env.observation_space
        if observation_space is None:
            raise RuntimeError("Environment does not have an observation space")
        self._env = env
        self.step = 0

    def process_action(self, action: GameAction) -> ActionResponse:
        with self._lock:
            obs = self._env.step(action)
            if obs is None:
                raise RuntimeError("Environment step returned None")

            self.step += 1
            reset_performed = False
            response = respond(obs.state, obs.frame, self.step)

            if obs.state == GameState.GAME_OVER:
                self._env.reset()
                reset_performed = True
                response = response.model_copy(update={"reset_performed": reset_performed})

            return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = GameRuntime()
    yield


app = FastAPI(title="arc3-agi", lifespan=lifespan)


@app.post("/action", response_model=ActionResponse)
def post_action(payload: ActionRequest, request: Request) -> ActionResponse:
    action = wait_for_action(payload.action)
    runtime = cast(GameRuntime, request.app.state.runtime)
    return runtime.process_action(action)

def main() -> int:
    host, port = resolve_api_bind_address()
    uvicorn.run("arc3_agi.script:app", host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())