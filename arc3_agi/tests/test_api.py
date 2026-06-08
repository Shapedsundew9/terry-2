from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from arc3_agi import script
from arcengine import GameAction, GameState


class FakeRuntime:
    def __init__(self) -> None:
        self.last_action: GameAction | None = None

    def process_action(self, action: GameAction) -> script.ActionResponse:
        self.last_action = action
        return script.ActionResponse(
            state=GameState.NOT_FINISHED.name,
            frame=[[[1, 2], [3, 4]]],
            step=1,
            reset_performed=False,
        )


def test_post_action_returns_state_and_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "GameRuntime", FakeRuntime)

    with TestClient(script.app) as client:
        response = client.post("/action", json={"action": "ACTION1"})

    assert response.status_code == 200
    assert response.json() == {
        "state": "NOT_FINISHED",
        "frame": [[[1, 2], [3, 4]]],
        "step": 1,
        "reset_performed": False,
    }


def test_post_action_accepts_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "GameRuntime", FakeRuntime)

    with TestClient(script.app) as client:
        response = client.post("/action", json={"action": "RESET"})

    assert response.status_code == 200
    assert response.json()["frame"] == [[[1, 2], [3, 4]]]


def test_post_action_rejects_invalid_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "GameRuntime", FakeRuntime)

    with TestClient(script.app) as client:
        response = client.post("/action", json={"action": "NOT_A_REAL_ACTION"})

    assert response.status_code == 400
    assert "Invalid action 'NOT_A_REAL_ACTION'" in response.json()["detail"]


def test_game_runtime_sets_reset_flag_and_resets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.observation_space = SimpleNamespace(frame=[[0]])
            self.reset_called = False

        def step(self, action: GameAction) -> SimpleNamespace:
            assert action == GameAction.ACTION1
            return SimpleNamespace(state=GameState.GAME_OVER, frame=[[[9]]])

        def reset(self) -> None:
            self.reset_called = True

    fake_env = FakeEnv()

    class FakeArcade:
        def make(self, game_id: str) -> FakeEnv:
            assert game_id == "ls20"
            return fake_env

    monkeypatch.setattr(script.arc_agi, "Arcade", FakeArcade)
    monkeypatch.setattr(script, "write_frame_to_json_file", lambda frame, output_path: None)

    runtime = script.GameRuntime()
    result = runtime.process_action(GameAction.ACTION1)

    assert result.state == GameState.GAME_OVER.name
    assert result.frame == [[[9]]]
    assert result.step == 1
    assert result.reset_performed is True
    assert fake_env.reset_called is True
