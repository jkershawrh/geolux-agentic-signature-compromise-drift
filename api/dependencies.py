from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from db.database import create_db_engine, get_session_factory, init_db
from db.repository import Repository
from engine.identity_pipeline import IdentityPipeline
from engine.secure_measurement import SecureMeasurement
from engine.signature_generator import SignatureGenerator

# Process-wide singletons: engine/session factory and the stateless (or
# expensive-to-build) pipeline components. Sessions are NOT shared —
# each request gets its own via the get_repo dependency, because a
# SQLAlchemy Session is not thread-safe and FastAPI serves sync
# endpoints from a threadpool.
_session_factory = None
_components: tuple | None = None
_discriminative_masks: dict[str, list[bool]] = {}


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        # create_db_engine(None) resolves ASC_DATABASE_PATH or falls back
        # to <repo>/data/asc.db; make sure the default directory exists.
        db_path = os.environ.get("ASC_DATABASE_PATH")
        if db_path is None:
            (Path(__file__).parent.parent / "data").mkdir(parents=True, exist_ok=True)
        engine = create_db_engine(db_path)
        init_db(engine)
        _session_factory = get_session_factory(engine)
    return _session_factory


def _get_components() -> tuple:
    global _components
    if _components is None:
        # NOTE: the API is wired to a mock inference adapter — enrollment,
        # certification, and monitoring run against simulated responses.
        # Swap in a real adapter (e.g. LiteLLMAdapter) for live inference.
        adapter = RealisticMockAdapter(profile="balanced")
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        secure = SecureMeasurement()
        _components = (adapter, extractor, generator, secure)
    return _components


def get_repo() -> Iterator[Repository]:
    """FastAPI dependency: a Repository bound to a per-request session."""
    session = _get_session_factory()()
    try:
        yield Repository(session)
    finally:
        session.close()


def get_pipeline(repo: Repository = Depends(get_repo)) -> IdentityPipeline:
    """FastAPI dependency: an IdentityPipeline bound to this request's repo."""
    adapter, extractor, generator, secure = _get_components()
    return IdentityPipeline(
        adapter=adapter,
        extractor=extractor,
        generator=generator,
        repository=repo,
        canary_threshold=0.1,
        secure=secure,
        discriminative_masks=_discriminative_masks,
    )
