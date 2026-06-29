#!/usr/bin/env python3
"""Cross-model embedding generalization study.

Tests whether embedding-based identity signatures work across different models.

Usage:
    python scripts/embedding_generalization.py             # Mock mode
    python scripts/embedding_generalization.py --maas      # Real MaaS
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from scripts.identity_validation import AGENT_DEFS, PROMPTS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 5 most differentiated agents (by ID)
KEY_AGENT_IDS = [
    "devops-sre",
    "code-reviewer",
    "customer-support",
    "clinical-advisor",
    "legal-advisor",
]

KEY_AGENT_DEFS = [d for d in AGENT_DEFS if d["id"] in KEY_AGENT_IDS]

# 4 GPU models
MODELS = [
    "granite-3-2-8b-instruct",
    "microsoft-phi-4",
    "llama-scout-17b",
    "deepseek-r1-distill-qwen-14b",
]

# First 10 prompts
STUDY_PROMPTS = PROMPTS[:10]

N_TRAIN = 5
N_TEST = 5
PCA_COMPONENTS = 20


# ---------------------------------------------------------------------------
# EER computation helper
# ---------------------------------------------------------------------------

def _compute_eer(genuine_dists: np.ndarray, impostor_dists: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate from genuine/impostor distance arrays.

    Returns (eer, eer_threshold).
    """
    if len(genuine_dists) == 0 or len(impostor_dists) == 0:
        return 0.5, 0.0

    max_dist = max(genuine_dists.max(), impostor_dists.max())
    thresholds = np.linspace(0, max_dist * 1.2, 300)

    far_curve = np.array([float(np.mean(impostor_dists < t)) for t in thresholds])
    frr_curve = np.array([float(np.mean(genuine_dists > t)) for t in thresholds])

    diff = np.abs(far_curve - frr_curve)
    eer_idx = int(np.argmin(diff))
    eer = float((far_curve[eer_idx] + frr_curve[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    agent_ids: list[str],
    response_texts: dict[str, list[str]],
    embedding_adapter,
) -> dict:
    """Build shared PCA from train embeddings, evaluate EER and accuracy.

    Returns a dict with eer, batch_accuracy, per_run_accuracy, raw embeddings,
    fitted PCA, and centroids for cross-model transfer testing.
    """
    # Embed all responses
    all_train_embeddings: list[np.ndarray] = []
    all_train_labels: list[str] = []
    test_embeddings: dict[str, list[np.ndarray]] = {aid: [] for aid in agent_ids}

    for aid in agent_ids:
        for idx, txt in enumerate(response_texts[aid]):
            emb = embedding_adapter.embed(txt)
            if idx < N_TRAIN:
                all_train_embeddings.append(emb)
                all_train_labels.append(aid)
            else:
                test_embeddings[aid].append(emb)

    all_emb_matrix = np.array(all_train_embeddings)
    n_comp = min(PCA_COMPONENTS, all_emb_matrix.shape[0] - 1, all_emb_matrix.shape[1])
    shared_pca = PCA(n_components=n_comp)
    all_projected = shared_pca.fit_transform(all_emb_matrix)

    # Build per-agent centroids in PCA space
    centroids: dict[str, np.ndarray] = {}
    idx = 0
    for aid in agent_ids:
        agent_projected = all_projected[idx:idx + N_TRAIN]
        centroids[aid] = agent_projected.mean(axis=0)
        idx += N_TRAIN

    # Evaluate batch accuracy
    batch_correct = 0
    for aid in agent_ids:
        test_embs = np.array(test_embeddings[aid])
        test_proj = shared_pca.transform(test_embs)
        test_centroid = test_proj.mean(axis=0)
        best_id = min(agent_ids, key=lambda x: euclidean_distance(test_centroid, centroids[x]))
        if best_id == aid:
            batch_correct += 1
    batch_accuracy = batch_correct / len(agent_ids) * 100

    # Evaluate per-run accuracy and collect genuine/impostor distances
    run_correct = 0
    run_total = 0
    genuine_dists: list[float] = []
    impostor_dists: list[float] = []

    for aid in agent_ids:
        test_embs = np.array(test_embeddings[aid])
        test_proj = shared_pca.transform(test_embs)
        for proj in test_proj:
            run_total += 1
            best_id = min(agent_ids, key=lambda x: euclidean_distance(proj, centroids[x]))
            if best_id == aid:
                run_correct += 1
            # Genuine distance
            genuine_dists.append(euclidean_distance(proj, centroids[aid]))
            # Impostor distances
            for other_id in agent_ids:
                if other_id == aid:
                    continue
                impostor_dists.append(euclidean_distance(proj, centroids[other_id]))

    per_run_accuracy = run_correct / run_total * 100 if run_total > 0 else 0.0
    eer, eer_threshold = _compute_eer(np.array(genuine_dists), np.array(impostor_dists))

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "batch_accuracy": batch_accuracy,
        "per_run_accuracy": per_run_accuracy,
        "pca": shared_pca,
        "centroids": centroids,
        "all_train_embeddings": all_emb_matrix,
        "all_train_labels": all_train_labels,
        "test_embeddings": test_embeddings,
        "explained_variance": float(sum(shared_pca.explained_variance_ratio_)) * 100,
    }


# ---------------------------------------------------------------------------
# Cross-model transfer test
# ---------------------------------------------------------------------------

def _cross_model_transfer(
    source_result: dict,
    target_result: dict,
    agent_ids: list[str],
) -> float:
    """Train PCA on source model data, evaluate on target model data.

    Projects target test embeddings into source PCA space and computes EER
    using source centroids to identify target responses.

    Returns EER.
    """
    source_pca = source_result["pca"]
    source_centroids = source_result["centroids"]
    target_test_embeddings = target_result["test_embeddings"]

    genuine_dists: list[float] = []
    impostor_dists: list[float] = []

    for aid in agent_ids:
        test_embs = np.array(target_test_embeddings[aid])
        # Project target embeddings into source PCA space
        test_proj = source_pca.transform(test_embs)
        for proj in test_proj:
            # Genuine distance
            genuine_dists.append(euclidean_distance(proj, source_centroids[aid]))
            # Impostor distances
            for other_id in agent_ids:
                if other_id == aid:
                    continue
                impostor_dists.append(euclidean_distance(proj, source_centroids[other_id]))

    eer, _ = _compute_eer(np.array(genuine_dists), np.array(impostor_dists))
    return eer


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_embedding_generalization(use_maas: bool = False) -> None:
    mode = "MaaS" if use_maas else "Mock"
    n_agents = len(KEY_AGENT_DEFS)
    n_prompts = len(STUDY_PROMPTS)
    n_models = len(MODELS)
    total_calls = n_agents * n_prompts * n_models  # chat calls
    total_embed = total_calls  # embedding calls
    total_all = total_calls + total_embed

    print("\n" + "#" * 60)
    print("  CROSS-MODEL EMBEDDING GENERALIZATION STUDY")
    print(f"  {n_agents} Agents | {n_prompts} Prompts | {n_models} Models | Mode: {mode}")
    print(f"  {total_calls} chat + {total_embed} embedding = {total_all} total calls")
    print("#" * 60)

    # ---------------------------------------------------------------
    # Build embedding adapter
    # ---------------------------------------------------------------
    if use_maas:
        from adapters.embedding_adapter import EmbeddingAdapter
        gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
        emb_adapter = EmbeddingAdapter(api_key=gpu_key)
    else:
        from adapters.embedding_adapter import MockEmbeddingAdapter
        emb_adapter = MockEmbeddingAdapter()

    # ---------------------------------------------------------------
    # Data collection: per model, per agent, per prompt
    # ---------------------------------------------------------------
    print("\n--- DATA COLLECTION ---")

    # model_id -> agent_id -> list[str] of response texts
    model_responses: dict[str, dict[str, list[str]]] = {}
    agent_ids = [d["id"] for d in KEY_AGENT_DEFS]

    for model_idx, model_id in enumerate(MODELS):
        print(f"\n  Model [{model_idx + 1}/{n_models}]: {model_id}")

        if use_maas:
            from adapters.litellm_adapter import LiteLLMAdapter
            gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
            adapter = LiteLLMAdapter(
                api_key=gpu_key,
                model_override=model_id,
                temperature=0.7,
            )
        else:
            from adapters.mock_adapter import RealisticMockAdapter
            # Use a different seed element per model to get different responses
            adapter = None  # built per agent below

        responses_for_model: dict[str, list[str]] = {}

        for agent_idx, defn in enumerate(KEY_AGENT_DEFS):
            aid = defn["id"]

            if use_maas:
                agent = AgentProfile(
                    agent_id=aid,
                    display_name=defn["name"],
                    model_id=model_id,
                    system_prompt=defn["system_prompt"],
                )
            else:
                profile = defn["mock_profile"]
                agent = AgentProfile(
                    agent_id=aid,
                    display_name=defn["name"],
                    model_id=f"mock-{profile}-{model_id}",
                    system_prompt=defn["system_prompt"],
                )
                adapter = RealisticMockAdapter(profile=profile)

            texts: list[str] = []
            for prompt in STUDY_PROMPTS:
                run = adapter.execute(agent, prompt)
                texts.append(run.response_text)

            responses_for_model[aid] = texts
            print(f"    [{agent_idx + 1}/{n_agents}] {defn['name']}: {len(texts)} responses")

        model_responses[model_id] = responses_for_model

    print(f"\n  Collected {n_models} models x {n_agents} agents x {n_prompts} prompts")
    print(f"  Split: {N_TRAIN} train / {N_TEST} test per agent")

    # ---------------------------------------------------------------
    # Per-model evaluation
    # ---------------------------------------------------------------
    print("\n--- PER-MODEL EVALUATION ---")

    model_results: dict[str, dict] = {}
    for model_id in MODELS:
        print(f"\n  Evaluating {model_id}...")
        result = _evaluate_model(agent_ids, model_responses[model_id], emb_adapter)
        model_results[model_id] = result
        print(f"    PCA: {result['pca'].n_components_} components, "
              f"{result['explained_variance']:.1f}% variance")
        print(f"    EER: {result['eer'] * 100:.1f}%  "
              f"Batch: {result['batch_accuracy']:.0f}%  "
              f"Per-run: {result['per_run_accuracy']:.0f}%")

    # ---------------------------------------------------------------
    # Cross-model comparison table
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  CROSS-MODEL EMBEDDING GENERALIZATION")
    print("=" * 65)

    # Truncate long model names for display
    def _short_model(m: str) -> str:
        if len(m) > 27:
            return m[:24] + "..."
        return m

    print(f"  {'Model':<27s}  {'EER':>5s}  {'Batch Acc':>9s}  {'Per-Run Acc':>10s}")
    line = "  " + chr(0x2500) * 27 + "  " + chr(0x2500) * 5 + "  " + chr(0x2500) * 9 + "  " + chr(0x2500) * 10
    print(line)
    for model_id in MODELS:
        r = model_results[model_id]
        print(f"  {_short_model(model_id):<27s}  {r['eer']*100:4.1f}%  {r['batch_accuracy']:8.0f}%  {r['per_run_accuracy']:9.0f}%")

    # ---------------------------------------------------------------
    # Cross-model transfer test
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  CROSS-MODEL TRANSFER TEST")
    print("=" * 65)
    print(f"  Train PCA + centroids on one model, evaluate on another.")

    source_model = MODELS[0]  # Granite
    transfer_results: dict[str, float] = {}

    print(f"\n  Source model: {source_model}")
    print()

    for target_model in MODELS[1:]:
        eer = _cross_model_transfer(
            model_results[source_model],
            model_results[target_model],
            agent_ids,
        )
        transfer_results[target_model] = eer
        print(f"  Train on {_short_model(source_model)} -> Test on {_short_model(target_model)}:    EER={eer*100:.1f}%")

    # Determine verdict
    transfer_eers = list(transfer_results.values())
    mean_transfer_eer = float(np.mean(transfer_eers))
    source_eer = model_results[source_model]["eer"]

    print()
    if mean_transfer_eer < 0.20:
        verdict = "PORTABLE -- Embedding fingerprints generalize across models"
    elif mean_transfer_eer < 0.35:
        verdict = "PARTIALLY PORTABLE -- Some degradation across models"
    else:
        verdict = "MODEL-SPECIFIC -- Fingerprints do not transfer well"
    print(f"  Mean transfer EER: {mean_transfer_eer*100:.1f}% (source EER: {source_eer*100:.1f}%)")
    print(f"  Verdict: {verdict}")

    # ---------------------------------------------------------------
    # Full transfer matrix (all source -> target combinations)
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  FULL TRANSFER MATRIX (EER %)")
    print("=" * 65)

    # Column headers
    short_models = [_short_model(m)[:12] for m in MODELS]
    col_label = "Source \\ Target"
    header = f"  {col_label:<16s}"
    for sm in short_models:
        header += f"  {sm:>12s}"
    print(header)
    print("  " + "-" * (16 + len(MODELS) * 14))

    for src_model in MODELS:
        row = f"  {_short_model(src_model)[:16]:<16s}"
        for tgt_model in MODELS:
            if src_model == tgt_model:
                # Self-EER (same model, no transfer)
                eer_val = model_results[src_model]["eer"]
                row += f"  {eer_val*100:11.1f}%"
            else:
                eer_val = _cross_model_transfer(
                    model_results[src_model],
                    model_results[tgt_model],
                    agent_ids,
                )
                row += f"  {eer_val*100:11.1f}%"
        print(row)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    eers = [model_results[m]["eer"] for m in MODELS]
    batch_accs = [model_results[m]["batch_accuracy"] for m in MODELS]
    run_accs = [model_results[m]["per_run_accuracy"] for m in MODELS]

    print(f"\n  Per-model EER range:       {min(eers)*100:.1f}% - {max(eers)*100:.1f}%")
    print(f"  Per-model batch acc range: {min(batch_accs):.0f}% - {max(batch_accs):.0f}%")
    print(f"  Per-model per-run range:   {min(run_accs):.0f}% - {max(run_accs):.0f}%")
    print(f"  Mean transfer EER:         {mean_transfer_eer*100:.1f}%")
    print(f"  Transfer verdict:          {verdict}")
    print()


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_embedding_generalization(use_maas)
