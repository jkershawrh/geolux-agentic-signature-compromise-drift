from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, HTTPException, Request

from api.routers import agents, monitor, reports


def require_api_key(request: Request) -> None:
    """Require X-API-Key when ASC_API_KEY is configured.

    When the environment variable is unset the API stays open (research
    default); set it for any deployment that leaves localhost.
    """
    expected = os.environ.get("ASC_API_KEY")
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Geolux Agent Identity API",
        description="Agent signature assignment, verification, and drift monitoring",
        version="0.1.0",
    )
    secured = [Depends(require_api_key)]
    app.include_router(agents.router, prefix="/agents", tags=["agents"], dependencies=secured)
    app.include_router(monitor.router, prefix="/monitor", tags=["monitor"], dependencies=secured)
    app.include_router(reports.router, prefix="/reports", tags=["reports"], dependencies=secured)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
