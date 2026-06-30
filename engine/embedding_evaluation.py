"""Offline evaluation helpers for embedding-space identity verification.

Used by research scripts and CI regression tests to compute EER / ROC-AUC /
batch accuracy with the same shared-PCA path as the product engine.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score

from engine.embedding_signature import EmbeddingSignatureGenerator
from engine.geometric.distance import euclidean_distance


# Pre-registered mock regression bounds (committed; not fitted to MaaS data).
# Guards the evaluation pipeline against math/regression bugs; MaaS numbers
# (3.6% EER, 0.992 AUC) remain validated offline via scripts/.
MOCK_MAX_EER: float = 0.20
MOCK_MIN_AUC: float = 0.90
MOCK_MIN_BATCH_ACCURACY: float = 0.90


def compute_roc_auc(genuine_dists: np.ndarray, impostor_dists: np.ndarray) -> float:
    """ROC AUC from distance distributions (lower distance = genuine)."""
    if len(genuine_dists) == 0 or len(impostor_dists) == 0:
        return 0.5
    labels = np.array([1] * len(genuine_dists) + [0] * len(impostor_dists))
    scores = np.concatenate([-np.asarray(genuine_dists), -np.asarray(impostor_dists)])
    return float(roc_auc_score(labels, scores))


def evaluate_shared_embedding_identity(
    agent_responses: dict[str, list[str]],
    gen: EmbeddingSignatureGenerator,
    *,
    n_train: int | None = None,
) -> dict:
    """Train/test split per agent; shared PCA on train; evaluate on held-out.

    Returns ``eer``, ``auc``, ``batch_accuracy``, ``per_run_accuracy``, and
    raw genuine/impostor distance arrays.
    """
    if len(agent_responses) < 2:
        raise ValueError("need at least two agents")
    n_prompts = min(len(v) for v in agent_responses.values())
    if n_prompts < 4:
        raise ValueError("need at least four responses per agent")
    n_train = n_train or max(2, n_prompts // 2)
    n_train = min(n_train, n_prompts - 1)

    train = {aid: texts[:n_train] for aid, texts in agent_responses.items()}
    test = {aid: texts[n_train:] for aid, texts in agent_responses.items()}

    gen.fit_shared(train)
    baselines = {aid: gen.generate_baseline(aid, train[aid]) for aid in agent_responses}

    genuine, impostor = [], []
    batch_correct = 0
    run_correct = 0
    run_total = 0

    for aid, texts in test.items():
        test_embs = [gen.project(t, baselines[aid]) for t in texts]
        test_centroid = np.mean(test_embs, axis=0)

        # batch: nearest centroid
        best_id = min(
            baselines,
            key=lambda x: euclidean_distance(test_centroid, np.array(baselines[x].centroid)),
        )
        if best_id == aid:
            batch_correct += 1

        for text in texts:
            proj = gen.project(text, baselines[aid])
            dist_own = euclidean_distance(proj, np.array(baselines[aid].centroid))
            genuine.append(dist_own)
            run_total += 1
            nearest = min(
                baselines,
                key=lambda x: euclidean_distance(proj, np.array(baselines[x].centroid)),
            )
            if nearest == aid:
                run_correct += 1
            for other_id, other_bl in baselines.items():
                if other_id == aid:
                    continue
                dist_imp = euclidean_distance(proj, np.array(other_bl.centroid))
                impostor.append(dist_imp)

    from engine.embedding_signature import compute_eer

    eer, _ = compute_eer(np.array(genuine), np.array(impostor))
    auc = compute_roc_auc(np.array(genuine), np.array(impostor))
    n_agents = len(agent_responses)
    return {
        "eer": eer,
        "auc": auc,
        "batch_accuracy": batch_correct / n_agents,
        "per_run_accuracy": run_correct / run_total if run_total else 0.0,
        "n_train": n_train,
        "n_test_per_agent": n_prompts - n_train,
        "genuine_dists": genuine,
        "impostor_dists": impostor,
    }
