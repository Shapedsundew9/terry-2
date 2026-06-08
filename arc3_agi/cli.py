"""Command line entry point for arc3-agi."""

from argparse import ArgumentParser

import uvicorn


def main() -> int:
    """Run the arc3-agi API server."""
    parser = ArgumentParser(description="Run the arc3-agi HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run("arc3_agi.script:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
