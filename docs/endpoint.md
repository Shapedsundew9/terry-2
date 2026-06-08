# Endpoint Overview

This service exposes one HTTP POST endpoint:

- Path: /action
- Purpose: Accept one game action, apply one environment step, and return the updated game state.

## Request

JSON body fields:

- action: string

The action must be a valid GameAction enum name (for example ACTION1 or RESET).

For initial-frame retrieval, clients should send action=RESET.

## Response

On success, the endpoint returns HTTP 200 with JSON:

- state: string
- frame: array of layers, where each layer is a 2D integer grid
- step: number
- reset_performed: boolean

Behavior notes:

- step increases by 1 for each accepted action.
- If state is GAME_OVER, the environment is reset automatically and reset_performed is true.
- REST clients that require startup state (like the Rust renderer) should fail fast if the API call fails or returns malformed frame data.
- For startup frame loading via RESET, clients should read frame[0] as the first layer.

## Error Handling

- Invalid action values return HTTP 400.
- The error message includes the list of valid action names.

## Quick Example

curl -X POST <http://127.0.0.1:8000/action> -H "Content-Type: application/json" -d '{"action":"RESET"}'
