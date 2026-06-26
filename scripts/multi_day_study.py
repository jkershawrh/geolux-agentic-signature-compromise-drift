#!/usr/bin/env python3
"""Multi-day study runner: 2000 API calls across 3 days.

Usage:
    python scripts/multi_day_study.py day1    # Cross-model validation (~650 calls)
    python scripts/multi_day_study.py day2    # Confidence intervals (~650 calls)
    python scripts/multi_day_study.py day3    # Large model exploration (~700 calls)
    python scripts/multi_day_study.py summary # Aggregate all study results
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from adapters.litellm_adapter import LiteLLMAdapter
from adapters.metric_extractor import DefaultMetricExtractor
from db.database import create_db_engine, get_session_factory, init_db
from db.repository import Repository
from domain.models import AgentProfile
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.reducibility_analyzer import ReducibilityAnalyzer
from scripts.identity_validation import AGENT_DEFS, PROMPTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FISHER_TOP_K = 6
TEMPERATURE = 0.7

# Day 1 models
DAY1_MODELS = [
    "microsoft-phi-4",
    "qwen3-14b",
    "granite-3-2-8b-instruct",
]

# Day 2: Granite 8B with different prompt slices
DAY2_MODEL = "granite-3-2-8b-instruct"
DAY2_PROMPT_SLICES = {
    "2a": PROMPTS[0:10],
    "2b": PROMPTS[5:15],
    "2c": PROMPTS[10:20],
}

# Day 3: 5 key agents on large models
DAY3_KEY_AGENT_IDS = [
    "devops-sre",
    "code-reviewer",
    "customer-support",
    "clinical-advisor",
    "legal-advisor",
]
DAY3_MODELS = [
    "gpt-oss-120b",
    "deepseek-r1-distill-qwen-14b",
    "llama-scout-17b",
]
DAY3_RUNS_PER_AGENT = 20

# Resolve key agent defs
DAY3_AGENT_DEFS = [d for d in AGENT_DEFS if d["id"] in DAY3_KEY_AGENT_IDS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_repo() -> Repository:
    """Create engine, init DB, return a Repository."""
    engine = create_db_engine()
    init_db(engine)
    factory = get_session_factory(engine)
    session = factory()
    return Repository(session)


def _get_gpu_adapter(model_id: str) -> LiteLLMAdapter:
    """Build a LiteLLMAdapter for a GPU model."""
    gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
    return LiteLLMAdapter(
        model_override=model_id,
        api_key=gpu_key,
        temperature=TEMPERATURE,
    )


def _make_agent(defn: dict, model_id: str) -> AgentProfile:
    """Build an AgentProfile from a definition dict."""
    return AgentProfile(
        agent_id=defn["id"],
        display_name=defn["name"],
        model_id=model_id,
        system_prompt=defn["system_prompt"],
    )


def _collect_agent_vectors(
    adapter: LiteLLMAdapter,
    agent: AgentProfile,
    prompts: list[str],
    study_id: str,
    agent_label: str,
) -> np.ndarray:
    """Run an agent on all prompts, return matrix of metric vectors.

    Prints progress as: [study_id] Agent X: run Y/Z
    """
    extractor = DefaultMetricExtractor()
    vectors = []
    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        run = adapter.execute(agent, prompt)
        vec = metrics_to_vector(extractor.extract(run))
        vectors.append(vec)
        print(f"  [{study_id}] {agent_label}: run {idx + 1}/{total}")
    return np.array(vectors)


def _compute_fisher_separation(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Compute Fisher top-k separation ratio between two agent matrices."""
    analyzer = ReducibilityAnalyzer()
    fisher_ratios = analyzer.compute_fisher_ratios(matrix_a, matrix_b)
    mask = analyzer.get_discriminative_mask(fisher_ratios, top_k=FISHER_TOP_K)
    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        return 0.0
    af = matrix_a[:, indices]
    bf = matrix_b[:, indices]
    within_a = [euclidean_distance(a, b) for a, b in itertools.combinations(af, 2)]
    within_b = [euclidean_distance(a, b) for a, b in itertools.combinations(bf, 2)]
    inter = [euclidean_distance(a, b) for a, b in itertools.product(af, bf)]
    within_all = within_a + within_b
    mean_within = float(np.mean(within_all)) if within_all else 0.0
    if mean_within == 0:
        return float("inf")
    return float(np.mean(inter) / mean_within)


def _compute_batch_accuracy(agent_data: dict[str, np.ndarray], agent_ids: list[str]) -> float:
    """Compute batch accuracy using first-half fingerprint, second-half verification."""
    from domain.metrics import ALL_METRIC_NAMES

    analyzer = ReducibilityAnalyzer()
    n_agents = len(agent_ids)
    half = agent_data[agent_ids[0]].shape[0] // 2
    if half < 2:
        half = 2

    # Compute global Fisher indices
    pairs = list(itertools.combinations(range(n_agents), 2))
    agg_fisher: dict[str, float] = {name: 0.0 for name in ALL_METRIC_NAMES}
    for ai, aj in pairs:
        ratios = analyzer.compute_fisher_ratios(agent_data[agent_ids[ai]], agent_data[agent_ids[aj]])
        for m, v in ratios.items():
            agg_fisher[m] += v / len(pairs)
    sorted_metrics = sorted(agg_fisher.items(), key=lambda x: -x[1])
    top_names = {name for name, _ in sorted_metrics[:FISHER_TOP_K]}
    global_fisher_indices = [i for i, name in enumerate(ALL_METRIC_NAMES) if name in top_names]

    # Fingerprint centroids from first half
    fingerprint_centroids = {}
    for aid in agent_ids:
        mat = agent_data[aid][:half, :][:, global_fisher_indices]
        fingerprint_centroids[aid] = mat.mean(axis=0)

    # Batch verification with second half
    batch_correct = 0
    for aid in agent_ids:
        test_mat = agent_data[aid][half:, :][:, global_fisher_indices]
        if test_mat.shape[0] == 0:
            continue
        test_centroid = test_mat.mean(axis=0)
        best_id = min(
            fingerprint_centroids,
            key=lambda x: euclidean_distance(test_centroid, fingerprint_centroids[x]),
        )
        if best_id == aid:
            batch_correct += 1
    return batch_correct / n_agents * 100


def _compute_eer(agent_data: dict[str, np.ndarray], agent_ids: list[str]) -> float:
    """Compute Equal Error Rate from agent data."""
    from domain.metrics import ALL_METRIC_NAMES

    analyzer = ReducibilityAnalyzer()
    n_agents = len(agent_ids)
    half = agent_data[agent_ids[0]].shape[0] // 2
    if half < 2:
        half = 2

    # Compute global Fisher indices
    pairs = list(itertools.combinations(range(n_agents), 2))
    agg_fisher: dict[str, float] = {name: 0.0 for name in ALL_METRIC_NAMES}
    for ai, aj in pairs:
        ratios = analyzer.compute_fisher_ratios(agent_data[agent_ids[ai]], agent_data[agent_ids[aj]])
        for m, v in ratios.items():
            agg_fisher[m] += v / len(pairs)
    sorted_metrics = sorted(agg_fisher.items(), key=lambda x: -x[1])
    top_names = {name for name, _ in sorted_metrics[:FISHER_TOP_K]}
    fisher_indices = [i for i, name in enumerate(ALL_METRIC_NAMES) if name in top_names]

    # Fingerprint centroids
    fingerprint_centroids = {}
    for aid in agent_ids:
        mat = agent_data[aid][:half, :][:, fisher_indices]
        fingerprint_centroids[aid] = mat.mean(axis=0)

    # Genuine and impostor distances
    genuine_dists = []
    impostor_dists = []
    for aid in agent_ids:
        test_vecs = agent_data[aid][half:, :][:, fisher_indices]
        fp = fingerprint_centroids[aid]
        for vec in test_vecs:
            genuine_dists.append(euclidean_distance(vec, fp))
        for other_aid in agent_ids:
            if other_aid == aid:
                continue
            other_test = agent_data[other_aid][half:, :][:, fisher_indices]
            for vec in other_test:
                impostor_dists.append(euclidean_distance(vec, fp))

    if not genuine_dists or not impostor_dists:
        return 0.5

    genuine_arr = np.array(genuine_dists)
    impostor_arr = np.array(impostor_dists)
    max_dist = max(genuine_arr.max(), impostor_arr.max())
    thresholds = np.linspace(0, max_dist * 1.2, 200)

    far_curve = np.array([float(np.mean(impostor_arr < t)) for t in thresholds])
    frr_curve = np.array([float(np.mean(genuine_arr > t)) for t in thresholds])

    diff = np.abs(far_curve - frr_curve)
    eer_idx = int(np.argmin(diff))
    return float((far_curve[eer_idx] + frr_curve[eer_idx]) / 2)


def _run_study(
    study_id: str,
    study_name: str,
    model_id: str,
    agent_defs: list[dict],
    prompts: list[str],
    repo: Repository,
) -> dict:
    """Run a full study: all agents on all prompts, compute metrics, save to DB.

    Returns a results dict with separation ratios, accuracy, EER, etc.
    """
    repo.save_study(
        study_id=study_id,
        study_name=study_name,
        model_id=model_id,
        agents_count=len(agent_defs),
        runs_per_agent=len(prompts),
    )

    try:
        adapter = _get_gpu_adapter(model_id)
        agent_data: dict[str, np.ndarray] = {}
        agent_ids = []

        for defn in agent_defs:
            agent = _make_agent(defn, model_id)
            aid = defn["id"]
            agent_ids.append(aid)
            agent_data[aid] = _collect_agent_vectors(
                adapter, agent, prompts, study_id, defn["name"],
            )

        # Compute pairwise Fisher separation ratios
        n_agents = len(agent_ids)
        pairs = list(itertools.combinations(range(n_agents), 2))
        pairwise_ratios: dict[str, float] = {}
        for i, j in pairs:
            ratio = _compute_fisher_separation(
                agent_data[agent_ids[i]], agent_data[agent_ids[j]],
            )
            pair_key = f"{agent_ids[i]}|{agent_ids[j]}"
            pairwise_ratios[pair_key] = ratio

        all_ratios = list(pairwise_ratios.values())
        mean_ratio = float(np.mean(all_ratios)) if all_ratios else 0.0
        pairs_above_3 = sum(1 for v in all_ratios if v > 3.0)

        # Batch accuracy
        batch_acc = _compute_batch_accuracy(agent_data, agent_ids)

        # EER
        eer = _compute_eer(agent_data, agent_ids)

        total_runs = len(agent_defs) * len(prompts)
        results = {
            "total_runs": total_runs,
            "mean_ratio": mean_ratio,
            "pairs_above_3": pairs_above_3,
            "total_pairs": len(pairs),
            "batch_accuracy": batch_acc,
            "eer": eer,
            "pairwise_ratios": pairwise_ratios,
        }
        repo.complete_study(study_id, results)

        print(f"\n  [{study_id}] COMPLETED: ratio={mean_ratio:.2f}, "
              f"pairs>3={pairs_above_3}/{len(pairs)}, "
              f"acc={batch_acc:.1f}%, EER={eer * 100:.1f}%")
        return results

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        repo.fail_study(study_id, error_msg)
        print(f"\n  [{study_id}] FAILED: {exc}")
        return {"error": str(exc), "total_runs": 0}


# ---------------------------------------------------------------------------
# Day 1: Cross-Model Validation (~650 calls)
# ---------------------------------------------------------------------------

def run_day1() -> None:
    """Cross-model validation: 15 agents x 10 prompts on 3 models."""
    print("\n" + "=" * 60)
    print("  DAY 1: Cross-Model Validation (~650 API calls)")
    print("=" * 60)

    repo = _get_repo()
    prompts = PROMPTS[:10]
    results_by_model: dict[str, dict] = {}

    for idx, model_id in enumerate(DAY1_MODELS):
        study_id = f"day1-{chr(ord('a') + idx)}-{model_id}"
        study_name = f"Day 1: Cross-model {model_id}"
        print(f"\n--- Study {study_id} ---")
        print(f"  Model: {model_id}")
        print(f"  Agents: {len(AGENT_DEFS)}, Prompts: {len(prompts)}")
        print(f"  Expected calls: {len(AGENT_DEFS) * len(prompts)}")

        results = _run_study(study_id, study_name, model_id, AGENT_DEFS, prompts, repo)
        results_by_model[model_id] = results

    # Print comparison table
    print("\n\n" + "=" * 60)
    print("  DAY 1: Cross-Model Comparison")
    print("=" * 60)
    print(f"  {'Model':<30s}  {'Mean Ratio':>10s}  {'Pairs>3.0':>10s}  {'Batch Acc':>10s}  {'EER':>8s}")
    print(f"  {'-' * 30}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    total_calls = 0
    for model_id, res in results_by_model.items():
        if "error" in res:
            print(f"  {model_id:<30s}  {'FAILED':>10s}")
        else:
            p3 = f"{res['pairs_above_3']}/{res['total_pairs']}"
            print(f"  {model_id:<30s}  {res['mean_ratio']:>10.2f}  {p3:>10s}  "
                  f"{res['batch_accuracy']:>9.1f}%  {res['eer'] * 100:>7.1f}%")
        total_calls += res.get("total_runs", 0)
    print(f"\n  Total API calls: {total_calls}")


# ---------------------------------------------------------------------------
# Day 2: Confidence Intervals (~650 calls)
# ---------------------------------------------------------------------------

def run_day2() -> None:
    """Confidence intervals: 15 agents on Granite 8B, 3 prompt slices."""
    print("\n" + "=" * 60)
    print("  DAY 2: Confidence Intervals (~650 API calls)")
    print("=" * 60)

    repo = _get_repo()
    results_list: list[dict] = []

    for label, prompts in DAY2_PROMPT_SLICES.items():
        study_id = f"day2-{label}-granite8b"
        study_name = f"Day 2: Confidence interval run {label}"
        print(f"\n--- Study {study_id} ---")
        print(f"  Model: {DAY2_MODEL}")
        print(f"  Prompts: {len(prompts)} (slice {label})")
        print(f"  Expected calls: {len(AGENT_DEFS) * len(prompts)}")

        results = _run_study(study_id, study_name, DAY2_MODEL, AGENT_DEFS, prompts, repo)
        results_list.append(results)

    # Compute confidence intervals
    valid_results = [r for r in results_list if "error" not in r]

    print("\n\n" + "=" * 60)
    print("  DAY 2: Confidence Interval Analysis")
    print("=" * 60)

    if len(valid_results) >= 2:
        ratios = [r["mean_ratio"] for r in valid_results]
        eers = [r["eer"] * 100 for r in valid_results]
        accs = [r["batch_accuracy"] for r in valid_results]

        print(f"\n  Separation ratio:  {np.mean(ratios):.2f} +/- {np.std(ratios):.2f}")
        print(f"  EER:               {np.mean(eers):.1f}% +/- {np.std(eers):.1f}%")
        print(f"  Batch accuracy:    {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")

        # Per-pair consistency analysis
        print("\n  Per-pair consistency (across runs):")
        all_pair_keys: set[str] = set()
        for r in valid_results:
            all_pair_keys.update(r.get("pairwise_ratios", {}).keys())

        consistent_above_3 = 0
        consistent_below_3 = 0
        inconsistent = 0
        for pk in sorted(all_pair_keys):
            values = []
            for r in valid_results:
                val = r.get("pairwise_ratios", {}).get(pk)
                if val is not None:
                    values.append(val)
            if len(values) >= 2:
                all_above = all(v > 3.0 for v in values)
                all_below = all(v <= 3.0 for v in values)
                if all_above:
                    consistent_above_3 += 1
                elif all_below:
                    consistent_below_3 += 1
                else:
                    inconsistent += 1

        total_checked = consistent_above_3 + consistent_below_3 + inconsistent
        print(f"    Always > 3.0:    {consistent_above_3}/{total_checked}")
        print(f"    Always <= 3.0:   {consistent_below_3}/{total_checked}")
        print(f"    Inconsistent:    {inconsistent}/{total_checked}")
    else:
        print("  Not enough successful runs for confidence interval analysis.")

    total_calls = sum(r.get("total_runs", 0) for r in results_list)
    print(f"\n  Total API calls: {total_calls}")


# ---------------------------------------------------------------------------
# Day 3: Large Model + Exploration (~700 calls)
# ---------------------------------------------------------------------------

def run_day3() -> None:
    """Large model exploration: 5 key agents on 3 large models + embedding experiment."""
    print("\n" + "=" * 60)
    print("  DAY 3: Large Model Exploration (~700 API calls)")
    print("=" * 60)

    repo = _get_repo()
    prompts = PROMPTS[:DAY3_RUNS_PER_AGENT]
    results_by_model: dict[str, dict] = {}

    # Studies 3A-3C: 5 agents x 20 runs on each large model
    for idx, model_id in enumerate(DAY3_MODELS):
        study_id = f"day3-{chr(ord('a') + idx)}-{model_id}"
        study_name = f"Day 3: Large model {model_id}"
        print(f"\n--- Study {study_id} ---")
        print(f"  Model: {model_id}")
        print(f"  Agents: {len(DAY3_AGENT_DEFS)}")
        print(f"  Runs per agent: {len(prompts)}")
        print(f"  Expected calls: {len(DAY3_AGENT_DEFS) * len(prompts)}")

        results = _run_study(
            study_id, study_name, model_id, DAY3_AGENT_DEFS, prompts, repo,
        )
        results_by_model[model_id] = results

    # Study 3D: Embedding exploration (experimental)
    study_id_embed = "day3-d-embedding-exploration"
    print(f"\n--- Study {study_id_embed} (Experimental) ---")
    print("  Attempting nomic-embed-text-v1-5 semantic comparison...")

    repo.save_study(
        study_id=study_id_embed,
        study_name="Day 3: Embedding exploration (nomic-embed-text-v1-5)",
        model_id="nomic-embed-text-v1-5",
        agents_count=len(DAY3_AGENT_DEFS),
        runs_per_agent=0,
    )

    try:
        # Try to use the embedding model for semantic comparison
        import requests as req

        base_url = os.environ.get("LITELLM_API_BASE", "").rstrip("/")
        gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if gpu_key:
            headers["Authorization"] = f"Bearer {gpu_key}"

        # Collect sample responses from key agents using granite
        adapter = _get_gpu_adapter("granite-3-2-8b-instruct")
        sample_responses: dict[str, list[str]] = {}
        for defn in DAY3_AGENT_DEFS:
            agent = _make_agent(defn, "granite-3-2-8b-instruct")
            responses = []
            for prompt in PROMPTS[:5]:
                run = adapter.execute(agent, prompt)
                responses.append(run.response_text or "")
            sample_responses[defn["id"]] = responses

        # Try embedding endpoint
        embed_results: dict[str, list[list[float]]] = {}
        for aid, responses in sample_responses.items():
            payload = {
                "model": "nomic-embed-text-v1-5",
                "input": responses,
            }
            resp = req.post(
                f"{base_url}/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            embed_results[aid] = embeddings

        # Compute pairwise cosine similarities between agent embeddings
        embed_sims: dict[str, float] = {}
        agent_ids = list(embed_results.keys())
        for i, j in itertools.combinations(range(len(agent_ids)), 2):
            emb_i = np.mean(embed_results[agent_ids[i]], axis=0)
            emb_j = np.mean(embed_results[agent_ids[j]], axis=0)
            cos_sim = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))
            pair_key = f"{agent_ids[i]}|{agent_ids[j]}"
            embed_sims[pair_key] = cos_sim

        embed_study_results = {
            "total_runs": len(DAY3_AGENT_DEFS) * 5,
            "embedding_cosine_similarities": embed_sims,
            "mean_cosine_similarity": float(np.mean(list(embed_sims.values()))),
        }
        repo.complete_study(study_id_embed, embed_study_results)
        print(f"  Embedding exploration completed. Mean cosine sim: "
              f"{embed_study_results['mean_cosine_similarity']:.4f}")

    except Exception as exc:
        error_msg = f"Embedding exploration failed: {exc}"
        repo.fail_study(study_id_embed, error_msg)
        print(f"  {error_msg}")

    # Print Day 3 summary
    print("\n\n" + "=" * 60)
    print("  DAY 3: Large Model Results")
    print("=" * 60)
    print(f"  {'Model':<35s}  {'Mean Ratio':>10s}  {'Batch Acc':>10s}  {'EER':>8s}")
    print(f"  {'-' * 35}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    total_calls = 0
    for model_id, res in results_by_model.items():
        if "error" in res:
            print(f"  {model_id:<35s}  {'FAILED':>10s}")
        else:
            print(f"  {model_id:<35s}  {res['mean_ratio']:>10.2f}  "
                  f"{res['batch_accuracy']:>9.1f}%  {res['eer'] * 100:>7.1f}%")
        total_calls += res.get("total_runs", 0)
    print(f"\n  Total API calls (Day 3): {total_calls}")


# ---------------------------------------------------------------------------
# Summary: Aggregate all study results
# ---------------------------------------------------------------------------

def run_summary() -> None:
    """Read all study records from DB and produce an aggregate report."""
    print("\n" + "=" * 60)
    print("  MULTI-DAY STUDY AGGREGATE RESULTS")
    print("=" * 60)

    repo = _get_repo()
    studies = repo.list_studies()

    if not studies:
        print("\n  No studies found in database.")
        return

    # Categorize studies by day
    day1_studies = [s for s in studies if s.study_id.startswith("day1-")]
    day2_studies = [s for s in studies if s.study_id.startswith("day2-")]
    day3_studies = [s for s in studies if s.study_id.startswith("day3-") and "embedding" not in s.study_id]
    embed_studies = [s for s in studies if "embedding" in s.study_id]

    total_api_calls = sum(s.total_runs for s in studies)

    # Cross-Model Comparison (Day 1)
    print("\nCross-Model Comparison:")
    print(f"  {'Model':<30s}  {'Mean Ratio':>10s}  {'Pairs>3.0':>10s}  {'Batch Acc':>10s}  {'EER':>8s}")
    print(f"  {'-' * 30}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for s in day1_studies:
        if s.status == "completed" and s.results_json:
            res = json.loads(s.results_json)
            p3 = f"{res.get('pairs_above_3', '?')}/{res.get('total_pairs', '?')}"
            mean_r = res.get("mean_ratio", 0)
            acc = res.get("batch_accuracy", 0)
            eer = res.get("eer", 0) * 100
            print(f"  {s.model_id:<30s}  {mean_r:>10.2f}  {p3:>10s}  {acc:>9.1f}%  {eer:>7.1f}%")
        else:
            print(f"  {s.model_id:<30s}  {'(' + s.status + ')':>10s}")

    # Confidence Intervals (Day 2)
    print(f"\nConfidence Intervals ({DAY2_MODEL}, {len(day2_studies)} runs):")
    valid_d2 = []
    for s in day2_studies:
        if s.status == "completed" and s.results_json:
            valid_d2.append(json.loads(s.results_json))

    if len(valid_d2) >= 2:
        ratios = [r["mean_ratio"] for r in valid_d2]
        eers = [r["eer"] * 100 for r in valid_d2]
        accs = [r["batch_accuracy"] for r in valid_d2]
        print(f"  Separation ratio:      {np.mean(ratios):.2f} +/- {np.std(ratios):.2f}")
        print(f"  EER:                   {np.mean(eers):.1f}% +/- {np.std(eers):.1f}%")
        print(f"  Batch accuracy:        {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")
    else:
        print("  Insufficient completed runs for confidence intervals.")

    # Large Model Results (Day 3)
    print("\nLarge Model Results:")
    for s in day3_studies:
        if s.status == "completed" and s.results_json:
            res = json.loads(s.results_json)
            mean_r = res.get("mean_ratio", 0)
            acc = res.get("batch_accuracy", 0)
            print(f"  {s.model_id + ':':<35s} ratio={mean_r:.2f}, acc={acc:.1f}%")
        else:
            print(f"  {s.model_id + ':':<35s} ({s.status})")

    # Embedding results
    if embed_studies:
        print("\nEmbedding Exploration:")
        for s in embed_studies:
            if s.status == "completed" and s.results_json:
                res = json.loads(s.results_json)
                mean_sim = res.get("mean_cosine_similarity", 0)
                print(f"  Mean cosine similarity: {mean_sim:.4f}")
            else:
                print(f"  Status: {s.status}")

    # Total
    print(f"\nTotal API calls used:    {total_api_calls}/2000")

    # All studies listing
    print(f"\nAll Studies ({len(studies)}):")
    print(f"  {'Study ID':<45s}  {'Status':<12s}  {'Runs':>6s}  {'Model':<30s}")
    print(f"  {'-' * 45}  {'-' * 12}  {'-' * 6}  {'-' * 30}")
    for s in studies:
        print(f"  {s.study_id:<45s}  {s.status:<12s}  {s.total_runs:>6d}  {s.model_id:<30s}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args and run the requested day or summary."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "day1":
        run_day1()
    elif command == "day2":
        run_day2()
    elif command == "day3":
        run_day3()
    elif command == "summary":
        run_summary()
    elif command in ("--help", "-h", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
