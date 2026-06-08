import json
import random
from pathlib import Path
from typing import Any, Sequence

from arcengine import GameAction, GameState
import arc_agi


def write_frame_to_json_file(frame: Sequence[Any], output_path: str | Path) -> None:
    """Write a FrameDataRaw.frame value to a JSON file."""
    serializable_frame = [layer.tolist() if hasattr(layer, "tolist") else layer for layer in frame]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(serializable_frame, file_handle, indent=2)

def wait_for_action() -> GameAction:
    """Simulate waiting for an action to be posted. In a real implementation, this would likely involve
    listening for user input or receiving data from an external source."""
    # For demonstration purposes, we'll randomly select an action from the GameAction enum.
    return random.choice(list(GameAction))

def respond(state: GameState, frame: Sequence[Any]) -> None:
    """Simulate sending a response to the post with the next frame's data. In a real implementation, this would likely involve
    sending data to an external service or updating a user interface."""
    print(f"Current Game State: {state}")
    print(f"Received Frame Data: {frame}")

def main() -> int:
    # Initialize the ARC-AGI-3 client
    arc = arc_agi.Arcade()

    # Create an environment with human rendering
    env = arc.make("ls20", render_mode="terminal-fast")
    if env is None:
        print("Failed to create environment")
        return 1
    if env.observation_space is None:
        print("Environment does not have an observation space")
        return 1
    frame = env.observation_space.frame

    # Optionally, write the initial frame to a JSON file
    write_frame_to_json_file(frame, "initial_frame.json")
    
    # Play the game
    step = 0

    # Wait for an action to be posted.
    while (action := wait_for_action()) != GameAction.RESET:
        # Perform the action (rendering happens automatically)
        obs = env.step(action)
        assert obs is not None, "Environment step returned None"
        respond(obs.state, obs.frame)
        step += 1

        # Check game state
        if obs and obs.state == GameState.WIN:
            print(f"Game won at step {step}!")
            break
        elif obs and obs.state == GameState.GAME_OVER:
            env.reset()

        # Send a response to the post with the next frames data.

    # Get and display scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        print(f"Final Score: {scorecard.score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())