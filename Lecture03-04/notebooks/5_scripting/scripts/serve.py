#!/usr/bin/env python
"""Deployment pipeline: start the FastAPI inference server.

Usage
-----
    python scripts/serve.py --checkpoint runs/best.pt
    python scripts/serve.py --checkpoint runs/best.pt --host 0.0.0.0 --port 8080 --top-k 3

    # Curl example after the server is running:
    #   curl -X POST http://localhost:8080/predict -F "file=@cat.jpg"
    # Health check:
    #   curl http://localhost:8080/health
"""
from __future__ import annotations

import argparse
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a ResNet-18 checkpoint via FastAPI")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--config",     default="config/base.yaml")
    p.add_argument("--host",       default=None,  help="Override config serving.host")
    p.add_argument("--port",       type=int, default=None, help="Override config serving.port")
    p.add_argument("--top-k",      type=int, default=5,    help="Number of top predictions to return")
    p.add_argument("--reload",     action="store_true",    help="Enable uvicorn auto-reload (dev only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    host   = args.host or cfg["serving"]["host"]
    port   = args.port or cfg["serving"]["port"]

    # Load the predictor before uvicorn starts so startup errors surface immediately
    from src.serving.app import init_predictor
    init_predictor(args.checkpoint, top_k=args.top_k)

    print(f"Starting server on http://{host}:{port}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  top-k      : {args.top_k}")
    print(f"  docs       : http://{host}:{port}/docs")

    import uvicorn
    from src.serving.app import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
