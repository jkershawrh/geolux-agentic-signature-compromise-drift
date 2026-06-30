"""CI regression guard for embedding identity metrics (EER / AUC / accuracy).

Pre-registered bounds in ``engine.embedding_evaluation`` (MOCK_* constants).
Uses ``SeparatingMockEmbeddingAdapter`` so the pipeline is exercised with
reliable agent separation offline; MaaS headline numbers remain in scripts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from adapters.embedding_adapter import SeparatingMockEmbeddingAdapter
from engine.embedding_evaluation import (
    MOCK_MAX_EER,
    MOCK_MIN_AUC,
    MOCK_MIN_BATCH_ACCURACY,
    evaluate_shared_embedding_identity,
)
from engine.embedding_signature import EmbeddingSignatureGenerator


def _agent_corpus() -> dict[str, list[str]]:
    agents = ("alpha", "beta", "gamma", "delta")
    return {
        agent: [
            f"agent:{agent} response {i} about topic with unique wording {i * 7}"
            for i in range(10)
        ]
        for agent in agents
    }


@pytest.mark.regression
def test_embedding_pipeline_meets_preregistered_mock_bounds():
    gen = EmbeddingSignatureGenerator(SeparatingMockEmbeddingAdapter(), n_components=12)
    result = evaluate_shared_embedding_identity(_agent_corpus(), gen, n_train=5)

    assert result["eer"] <= MOCK_MAX_EER, (
        f"EER {result['eer']:.3f} exceeds mock bound {MOCK_MAX_EER}; "
        "shared PCA evaluation pipeline may be broken"
    )
    assert result["auc"] >= MOCK_MIN_AUC, (
        f"AUC {result['auc']:.3f} below mock bound {MOCK_MIN_AUC}"
    )
    assert result["batch_accuracy"] >= MOCK_MIN_BATCH_ACCURACY, (
        f"batch accuracy {result['batch_accuracy']:.3f} below {MOCK_MIN_BATCH_ACCURACY}"
    )
