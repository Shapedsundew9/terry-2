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
    
    # Exit for now
    return 0

    # Play the game
    for step in range(100):
        # Choose a random action
        action = random.choice(env.action_space)
        action_data = {}
        if action.is_complex():
            action_data = {
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            }

        # Perform the action (rendering happens automatically)
        obs = env.step(action, data=action_data)

        # Check game state
        if obs and obs.state == GameState.WIN:
            print(f"Game won at step {step}!")
            break
        elif obs and obs.state == GameState.GAME_OVER:
            env.reset()

    # Get and display scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        print(f"Final Score: {scorecard.score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())