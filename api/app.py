from __future__ import annotations

from fastapi import FastAPI

from api.routers import agents, monitor, reports


def create_app() -> FastAPI:
    app = FastAPI(
        title="Geolux Agent Identity API",
        description="Agent signature assignment, verification, and drift monitoring",
        version="0.1.0",
    )
    app.include_router(agents.router, prefix="/agents", tags=["agents"])
    app.include_router(monitor.router, prefix="/monitor", tags=["monitor"])
    app.include_router(reports.router, prefix="/reports", tags=["reports"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
