from __future__ import annotations

import os
from pathlib import Path

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from db.database import create_db_engine, init_db, get_session_factory
from db.repository import Repository
from engine.identity_pipeline import IdentityPipeline
from engine.signature_generator import SignatureGenerator

# Singletons
_pipeline = None
_repo = None


def get_pipeline():
    global _pipeline, _repo
    if _pipeline is None:
        adapter = RealisticMockAdapter(profile="balanced")
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")

        db_path = str(Path(__file__).parent.parent / "data" / "asc.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_db_engine(db_path)
        init_db(engine)
        session = get_session_factory(engine)()
        _repo = Repository(session)

        _pipeline = IdentityPipeline(
            adapter=adapter,
            extractor=extractor,
            generator=generator,
            repository=_repo,
            canary_threshold=0.1,
        )
    return _pipeline


def get_repo():
    get_pipeline()  # ensure initialized
    return _repo
