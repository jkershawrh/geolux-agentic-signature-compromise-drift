#!/usr/bin/env python3
"""Embedding-space signature validation: combined metric + embedding verification.

Usage:
    python scripts/embedding_validation.py             # Mock mode
    python scripts/embedding_validation.py --maas      # Real MaaS
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.embedding_signature import EmbeddingSignatureGenerator
from engine.reducibility_analyzer import ReducibilityAnalyzer
from scripts.identity_validation import (
    AGENT_DEFS,
    HARD_PAIRS,
    PROMPTS,
    FISHER_TOP_K,
    PHASE1_MODEL,
    _build_adapter_and_agent,
    _collect_agent_data,
)


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
# Main validation
# ---------------------------------------------------------------------------

def run_embedding_validation(use_maas: bool = False) -> None:
    mode = "MaaS" if use_maas else "Mock"
    print("\n" + "#" * 60)
    print("  EMBEDDING-SPACE SIGNATURE VALIDATION")
    print(f"  15 Agents | 10 Prompts | Mode: {mode}")
    print("#" * 60)

    # ---------------------------------------------------------------
    # Build adapters
    # ---------------------------------------------------------------
    if use_maas:
        from adapters.embedding_adapter import EmbeddingAdapter
        embedding_adapter = EmbeddingAdapter(
            api_key=os.environ.get("LITELLM_GPU_API_KEY", ""),
        )
    else:
        from adapters.embedding_adapter import MockEmbeddingAdapter
        embedding_adapter = MockEmbeddingAdapter()

    emb_gen = EmbeddingSignatureGenerator(embedding_adapter, n_components=20)

    # Use first 10 prompts for this study
    prompts = PROMPTS[:10]

    # Use the 15 base AGENT_DEFS
    agent_defs = AGENT_DEFS

    # Metric infrastructure
    metric_embedding_adapter = None
    if use_maas:
        try:
            from adapters.embedding_adapter import EmbeddingAdapter as EA
            metric_embedding_adapter = EA(
                api_key=os.environ.get("LITELLM_GPU_API_KEY", ""),
            )
        except Exception:
            pass
    extractor = DefaultMetricExtractor(embedding_adapter=metric_embedding_adapter)
    analyzer = ReducibilityAnalyzer()

    # ---------------------------------------------------------------
    # Phase 1: Data collection — 15 agents x 10 prompts
    # ---------------------------------------------------------------
    print("\n--- DATA COLLECTION ---")
    agent_ids = []
    short_names = {}
    metric_data = {}    # agent_id -> (n_prompts, n_metrics) array
    response_texts = {} # agent_id -> list[str]

    for i, defn in enumerate(agent_defs):
        adapter, agent = _build_adapter_and_agent(defn, use_maas)
        aid = defn["id"]
        agent_ids.append(aid)
        short_names[aid] = defn["short"]

        print(f"  [{i+1}/{len(agent_defs)}] Collecting {defn['name']}...")

        vecs = []
        texts = []
        for prompt in prompts:
            run = adapter.execute(agent, prompt)
            vec = metrics_to_vector(extractor.extract(run))
            vecs.append(vec)
            texts.append(run.response_text)

        metric_data[aid] = np.array(vecs)
        response_texts[aid] = texts

    n_prompts = len(prompts)
    n_train = 5
    n_test = n_prompts - n_train

    print(f"\n  Collected {len(agent_ids)} agents x {n_prompts} prompts")
    print(f"  Split: {n_train} train / {n_test} test")

    # ---------------------------------------------------------------
    # Phase 2: Compute global Fisher indices for metric baselines
    # ---------------------------------------------------------------
    print("\n--- BUILDING BASELINES ---")
    from domain.metrics import ALL_METRIC_NAMES

    agg_fisher = {name: 0.0 for name in ALL_METRIC_NAMES}
    pairs = list(itertools.combinations(agent_ids, 2))
    for ai, aj in pairs:
        ratios = analyzer.compute_fisher_ratios(metric_data[ai], metric_data[aj])
        for m, v in ratios.items():
            agg_fisher[m] += v / len(pairs)
    sorted_metrics = sorted(agg_fisher.items(), key=lambda x: -x[1])
    top_names = {name for name, _ in sorted_metrics[:FISHER_TOP_K]}
    fisher_indices = [i for i, name in enumerate(ALL_METRIC_NAMES) if name in top_names]

    print(f"  Top-{FISHER_TOP_K} Fisher metrics: {[name for name, _ in sorted_metrics[:FISHER_TOP_K]]}")

    # ---------------------------------------------------------------
    # Phase 3: Build metric + embedding baselines from train split
    # ---------------------------------------------------------------
    metric_centroids = {}
    metric_train_dists = {}
    embedding_train_dists = {}

    for aid in agent_ids:
        train_metric = metric_data[aid][:n_train, :][:, fisher_indices]
        metric_centroids[aid] = train_metric.mean(axis=0)
        dists = [euclidean_distance(v, metric_centroids[aid]) for v in train_metric]
        metric_train_dists[aid] = dists

    # Shared PCA for embeddings — fit on ALL agents' train embeddings together
    # so all agents are projected into the same space
    from sklearn.decomposition import PCA

    all_train_embeddings = []
    all_train_labels = []
    for aid in agent_ids:
        train_texts = response_texts[aid][:n_train]
        for t in train_texts:
            emb = emb_gen._adapter.embed(t)
            all_train_embeddings.append(emb)
            all_train_labels.append(aid)

    all_emb_matrix = np.array(all_train_embeddings)
    n_comp = min(20, all_emb_matrix.shape[0] - 1, all_emb_matrix.shape[1])
    shared_pca = PCA(n_components=n_comp)
    all_projected = shared_pca.fit_transform(all_emb_matrix)
    print(f"  Shared PCA: {n_comp} components, {sum(shared_pca.explained_variance_ratio_)*100:.1f}% variance")

    # Build per-agent embedding centroids in the shared PCA space
    embedding_centroids = {}
    idx = 0
    for aid in agent_ids:
        n = n_train
        agent_projected = all_projected[idx:idx+n]
        embedding_centroids[aid] = agent_projected.mean(axis=0)
        emb_dists = [euclidean_distance(v, embedding_centroids[aid]) for v in agent_projected]
        embedding_train_dists[aid] = emb_dists
        idx += n

    # Store shared PCA for projecting test data
    _shared_pca = shared_pca

    # Per-agent normalization stats
    metric_stats = {}  # aid -> (mean, std)
    embedding_stats = {}
    for aid in agent_ids:
        m_mean = float(np.mean(metric_train_dists[aid]))
        m_std = float(np.std(metric_train_dists[aid])) + 1e-10
        metric_stats[aid] = (m_mean, m_std)

        e_mean = float(np.mean(embedding_train_dists[aid]))
        e_std = float(np.std(embedding_train_dists[aid])) + 1e-10
        embedding_stats[aid] = (e_mean, e_std)

    print(f"  Built {len(metric_centroids)} metric centroids (dim={len(fisher_indices)})")
    print(f"  Built {len(embedding_centroids)} embedding centroids (dim={n_comp})")

    # ---------------------------------------------------------------
    # Phase 4: Evaluate — metric-only, embedding-only, combined
    # ---------------------------------------------------------------
    print("\n--- EVALUATION ---")

    # Collect genuine/impostor distances for all three modes
    metric_genuine = []
    metric_impostor = []
    embed_genuine = []
    embed_impostor = []
    combined_genuine = []
    combined_impostor = []

    metric_correct_batch = 0
    embed_correct_batch = 0
    combined_correct_batch = 0

    metric_correct_run = 0
    embed_correct_run = 0
    combined_correct_run = 0
    total_runs = 0

    for aid in agent_ids:
        test_metric_vecs = metric_data[aid][n_train:, :][:, fisher_indices]
        test_texts = response_texts[aid][n_train:]

        # -- Batch evaluation (centroid of test) --
        test_metric_centroid = test_metric_vecs.mean(axis=0)

        # Metric-only batch
        best_metric_id = min(
            agent_ids,
            key=lambda x: euclidean_distance(test_metric_centroid, metric_centroids[x]),
        )
        if best_metric_id == aid:
            metric_correct_batch += 1

        # Embedding-only batch: embed test texts, project via shared PCA
        test_projs = []
        for t in test_texts:
            emb = emb_gen._adapter.embed(t)
            proj = _shared_pca.transform(emb.reshape(1, -1))[0]
            test_projs.append(proj)
        test_emb_centroid = np.mean(test_projs, axis=0)
        best_embed_id = min(
            agent_ids,
            key=lambda x: euclidean_distance(
                test_emb_centroid, embedding_centroids[x]
            ),
        )
        if best_embed_id == aid:
            embed_correct_batch += 1

        # Combined batch
        best_combined_id = None
        best_combined_score = float("inf")
        for cand_id in agent_ids:
            m_dist = euclidean_distance(test_metric_centroid, metric_centroids[cand_id])
            e_dist = euclidean_distance(
                test_emb_centroid, embedding_centroids[cand_id]
            )
            m_z = (m_dist - metric_stats[cand_id][0]) / metric_stats[cand_id][1]
            e_z = (e_dist - embedding_stats[cand_id][0]) / embedding_stats[cand_id][1]
            combined_score = 0.5 * m_z + 0.5 * e_z
            if combined_score < best_combined_score:
                best_combined_score = combined_score
                best_combined_id = cand_id
        if best_combined_id == aid:
            combined_correct_batch += 1

        # -- Per-run evaluation --
        for run_idx in range(n_test):
            vec = test_metric_vecs[run_idx]
            txt = test_texts[run_idx]
            total_runs += 1

            # Metric-only per-run
            best_m = min(
                agent_ids,
                key=lambda x: euclidean_distance(vec, metric_centroids[x]),
            )
            m_dist_own = euclidean_distance(vec, metric_centroids[aid])
            if best_m == aid:
                metric_correct_run += 1

            # Embedding-only per-run (shared PCA space)
            emb_raw = emb_gen._adapter.embed(txt)
            proj = _shared_pca.transform(emb_raw.reshape(1, -1))[0]
            best_e = min(
                agent_ids,
                key=lambda x: euclidean_distance(proj, embedding_centroids[x]),
            )
            e_dist_own = euclidean_distance(proj, embedding_centroids[aid])
            if best_e == aid:
                embed_correct_run += 1

            # Combined per-run
            best_c = None
            best_c_score = float("inf")
            for cand_id in agent_ids:
                md = euclidean_distance(vec, metric_centroids[cand_id])
                ed = euclidean_distance(proj, embedding_centroids[cand_id])
                mz = (md - metric_stats[cand_id][0]) / metric_stats[cand_id][1]
                ez = (ed - embedding_stats[cand_id][0]) / embedding_stats[cand_id][1]
                cs = 0.5 * mz + 0.5 * ez
                if cs < best_c_score:
                    best_c_score = cs
                    best_c = cand_id
            if best_c == aid:
                combined_correct_run += 1

            # Genuine distances
            metric_genuine.append(m_dist_own)
            embed_genuine.append(e_dist_own)
            m_z_own = (m_dist_own - metric_stats[aid][0]) / metric_stats[aid][1]
            e_z_own = (e_dist_own - embedding_stats[aid][0]) / embedding_stats[aid][1]
            combined_genuine.append(0.5 * m_z_own + 0.5 * e_z_own)

            # Impostor distances: this run against all other agents
            for other_id in agent_ids:
                if other_id == aid:
                    continue
                m_imp = euclidean_distance(vec, metric_centroids[other_id])
                e_imp = euclidean_distance(proj, embedding_centroids[other_id])
                metric_impostor.append(m_imp)
                embed_impostor.append(e_imp)
                m_z_imp = (m_imp - metric_stats[other_id][0]) / metric_stats[other_id][1]
                e_z_imp = (e_imp - embedding_stats[other_id][0]) / embedding_stats[other_id][1]
                combined_impostor.append(0.5 * m_z_imp + 0.5 * e_z_imp)

    # Convert to arrays
    metric_genuine = np.array(metric_genuine)
    metric_impostor = np.array(metric_impostor)
    embed_genuine = np.array(embed_genuine)
    embed_impostor = np.array(embed_impostor)
    combined_genuine = np.array(combined_genuine)
    combined_impostor = np.array(combined_impostor)

    # Compute EER for each mode
    metric_eer, metric_eer_t = _compute_eer(metric_genuine, metric_impostor)
    embed_eer, embed_eer_t = _compute_eer(embed_genuine, embed_impostor)
    combined_eer, combined_eer_t = _compute_eer(combined_genuine, combined_impostor)

    # Compute Fisher ratios (mean across all pairs)
    all_ratios = list(itertools.combinations(agent_ids, 2))
    metric_fisher_vals = []
    for ai, aj in all_ratios:
        mat_i = metric_data[ai][:, fisher_indices]
        mat_j = metric_data[aj][:, fisher_indices]
        within_i = [euclidean_distance(a, b) for a, b in itertools.combinations(mat_i, 2)]
        within_j = [euclidean_distance(a, b) for a, b in itertools.combinations(mat_j, 2)]
        inter = [euclidean_distance(a, b) for a, b in itertools.product(mat_i, mat_j)]
        within_all = within_i + within_j
        mean_within = np.mean(within_all) if within_all else 0.0
        ratio = float(np.mean(inter) / mean_within) if mean_within > 0 else float("inf")
        metric_fisher_vals.append(ratio)
    mean_metric_fisher = float(np.mean(metric_fisher_vals))

    n_agents = len(agent_ids)
    batch_metric_acc = metric_correct_batch / n_agents * 100
    batch_embed_acc = embed_correct_batch / n_agents * 100
    batch_combined_acc = combined_correct_batch / n_agents * 100
    run_metric_acc = metric_correct_run / total_runs * 100 if total_runs > 0 else 0.0
    run_embed_acc = embed_correct_run / total_runs * 100 if total_runs > 0 else 0.0
    run_combined_acc = combined_correct_run / total_runs * 100 if total_runs > 0 else 0.0

    # ---------------------------------------------------------------
    # Phase 5: Hard pair analysis
    # ---------------------------------------------------------------
    print("\n--- HARD PAIR ANALYSIS ---")
    hard_pair_defs = [
        ("compliance-officer", "legal-advisor", "Compliance vs Legal"),
        ("support-a", "support-b", "Support A vs B"),
    ]

    # We need to collect data for hard-pair agents that may not be in agent_ids
    # support-a and support-b are from HARD_PAIRS, so collect them
    hp_metric_data = {}
    hp_response_texts = {}
    hp_ids_needed = set()
    for aid_a, aid_b, _ in hard_pair_defs:
        if aid_a not in metric_data:
            hp_ids_needed.add(aid_a)
        if aid_b not in metric_data:
            hp_ids_needed.add(aid_b)

    for defn in HARD_PAIRS:
        if defn["id"] in hp_ids_needed:
            adapter, agent = _build_adapter_and_agent(defn, use_maas)
            aid = defn["id"]
            print(f"  Collecting hard-pair agent {defn['name']}...")
            vecs = []
            texts = []
            for prompt in prompts:
                run = adapter.execute(agent, prompt)
                vec = metrics_to_vector(extractor.extract(run))
                vecs.append(vec)
                texts.append(run.response_text)
            hp_metric_data[aid] = np.array(vecs)
            hp_response_texts[aid] = texts

    # Merge
    all_metric = {**metric_data, **hp_metric_data}
    all_texts = {**response_texts, **hp_response_texts}

    hard_pair_results = {}
    for aid_a, aid_b, label in hard_pair_defs:
        if aid_a not in all_metric or aid_b not in all_metric:
            print(f"  Skipping {label}: missing data")
            continue

        mat_a = all_metric[aid_a][:, fisher_indices]
        mat_b = all_metric[aid_b][:, fisher_indices]

        # Metric separation
        within_a = [euclidean_distance(a, b) for a, b in itertools.combinations(mat_a, 2)]
        within_b = [euclidean_distance(a, b) for a, b in itertools.combinations(mat_b, 2)]
        inter = [euclidean_distance(a, b) for a, b in itertools.product(mat_a, mat_b)]
        within_all = within_a + within_b
        mean_within_m = np.mean(within_all) if within_all else 0.0
        metric_ratio = float(np.mean(inter) / mean_within_m) if mean_within_m > 0 else float("inf")

        # Embedding separation (shared PCA space)
        texts_a = all_texts[aid_a]
        texts_b = all_texts[aid_b]
        embs_a = np.array([emb_gen._adapter.embed(t) for t in texts_a])
        embs_b = np.array([emb_gen._adapter.embed(t) for t in texts_b])
        proj_a = _shared_pca.transform(embs_a)
        proj_b = _shared_pca.transform(embs_b)
        centroid_a_emb = proj_a.mean(axis=0)
        centroid_b_emb = proj_b.mean(axis=0)
        emb_dist = euclidean_distance(centroid_a_emb, centroid_b_emb)

        # Embedding separation ratio
        within_a_emb = [euclidean_distance(a, b) for a, b in itertools.combinations(proj_a, 2)]
        within_b_emb = [euclidean_distance(a, b) for a, b in itertools.combinations(proj_b, 2)]
        inter_emb = [euclidean_distance(a, b) for a, b in itertools.product(proj_a, proj_b)]
        within_all_emb = within_a_emb + within_b_emb
        mean_within_emb = np.mean(within_all_emb) if within_all_emb else 0.0
        emb_ratio = float(np.mean(inter_emb) / mean_within_emb) if mean_within_emb > 0 else 0.0

        m_centroid_a = mat_a.mean(axis=0)
        m_centroid_b = mat_b.mean(axis=0)
        m_sep = euclidean_distance(m_centroid_a, m_centroid_b)

        hard_pair_results[label] = {
            "metric": metric_ratio,
            "embedding": emb_ratio,
            "combined": float(m_sep + emb_dist) / 2,
        }

        print(f"  {label}:  metric={metric_ratio:.2f}  embedding={emb_dist:.2f}  combined={(m_sep + emb_dist)/2:.2f}")

    # ---------------------------------------------------------------
    # Phase 6: Report
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  EMBEDDING VALIDATION RESULTS")
    print("=" * 65)
    print(f"  {'':25s} {'Metric-Only':>14s} {'Embedding-Only':>16s} {'Combined':>10s}")
    print(f"  {'':25s} {'----------':>14s} {'--------------':>16s} {'--------':>10s}")
    print(f"  {'Mean Fisher ratio:':25s} {mean_metric_fisher:>13.2f}  {'N/A':>15s}  {'N/A':>9s}")
    print(f"  {'Batch accuracy:':25s} {batch_metric_acc:>12.0f}%  {batch_embed_acc:>14.0f}%  {batch_combined_acc:>8.0f}%")
    print(f"  {'Per-run accuracy:':25s} {run_metric_acc:>12.0f}%  {run_embed_acc:>14.0f}%  {run_combined_acc:>8.0f}%")
    print(f"  {'EER:':25s} {metric_eer*100:>12.1f}%  {embed_eer*100:>14.1f}%  {combined_eer*100:>8.1f}%")

    if hard_pair_results:
        print(f"\n  Hard Pairs:")
        for label, res in hard_pair_results.items():
            print(f"    {label:25s}  metric={res['metric']:.2f}  embedding={res['embedding']:.2f}  combined={res['combined']:.2f}")

    # Weight sweep: find optimal fusion weight
    print("\n  Fusion Weight Sweep (w=embedding weight):")
    print(f"  {'w':>5s} {'EER':>8s} {'Per-Run':>8s}")
    print(f"  {'─'*5} {'─'*8} {'─'*8}")
    best_w = 0.5
    best_w_eer = combined_eer
    for w_int in range(0, 11):
        w = w_int / 10.0
        sweep_genuine = []
        sweep_impostor = []
        sweep_correct = 0
        sweep_total = 0
        for aid in agent_ids:
            test_metric_vecs = metric_data[aid][n_train:, :][:, fisher_indices]
            test_texts = response_texts[aid][n_train:]
            for run_idx in range(n_test):
                vec = test_metric_vecs[run_idx]
                txt = test_texts[run_idx]
                emb_raw = emb_gen._adapter.embed(txt)
                proj = _shared_pca.transform(emb_raw.reshape(1, -1))[0]
                sweep_total += 1
                # Find nearest by weighted score
                best_cand = None
                best_score = float("inf")
                for cand_id in agent_ids:
                    md = euclidean_distance(vec, metric_centroids[cand_id])
                    ed = euclidean_distance(proj, embedding_centroids[cand_id])
                    mz = (md - metric_stats[cand_id][0]) / metric_stats[cand_id][1]
                    ez = (ed - embedding_stats[cand_id][0]) / embedding_stats[cand_id][1]
                    score = (1 - w) * mz + w * ez
                    if score < best_score:
                        best_score = score
                        best_cand = cand_id
                if best_cand == aid:
                    sweep_correct += 1
                # Genuine/impostor for EER
                m_d = euclidean_distance(vec, metric_centroids[aid])
                e_d = euclidean_distance(proj, embedding_centroids[aid])
                m_z = (m_d - metric_stats[aid][0]) / metric_stats[aid][1]
                e_z = (e_d - embedding_stats[aid][0]) / embedding_stats[aid][1]
                sweep_genuine.append((1 - w) * m_z + w * e_z)
                for other in agent_ids:
                    if other == aid:
                        continue
                    m_i = euclidean_distance(vec, metric_centroids[other])
                    e_i = euclidean_distance(proj, embedding_centroids[other])
                    mzi = (m_i - metric_stats[other][0]) / metric_stats[other][1]
                    ezi = (e_i - embedding_stats[other][0]) / embedding_stats[other][1]
                    sweep_impostor.append((1 - w) * mzi + w * ezi)
        # Compute EER at this weight
        sg = np.array(sweep_genuine)
        si = np.array(sweep_impostor)
        thresholds = np.linspace(min(sg.min(), si.min()), max(sg.max(), si.max()), 200)
        best_diff = float('inf')
        w_eer = 0.5
        for t in thresholds:
            far = float(np.mean(si <= t))
            frr = float(np.mean(sg > t))
            if abs(far - frr) < best_diff:
                best_diff = abs(far - frr)
                w_eer = (far + frr) / 2
        acc = sweep_correct / max(sweep_total, 1) * 100
        print(f"  {w:5.1f} {w_eer*100:7.1f}% {acc:7.0f}%")
        if w_eer < best_w_eer:
            best_w_eer = w_eer
            best_w = w

    print(f"\n  Optimal weight: {best_w:.1f} (EER={best_w_eer*100:.1f}%)")
    print(f"  Embedding-only (w=1.0): EER={embed_eer*100:.1f}%")
    print(f"  Metric-only (w=0.0): EER={metric_eer*100:.1f}%")

    # ---------------------------------------------------------------
    # PCA Component Sweep
    # ---------------------------------------------------------------
    print("\n  PCA Component Sweep:")
    print(f"  {'n_comp':>8s} {'EER':>8s} {'Per-Run':>8s}")
    for n_comp_target in [5, 10, 15, 20, 30, 50]:
        n_comp_actual = min(n_comp_target, all_emb_matrix.shape[0] - 1, all_emb_matrix.shape[1])
        pca_sweep = PCA(n_components=n_comp_actual)
        all_proj_sweep = pca_sweep.fit_transform(all_emb_matrix)
        # Build per-agent centroids in this PCA space
        sweep_centroids = {}
        sidx = 0
        for aid in agent_ids:
            agent_proj = all_proj_sweep[sidx:sidx + n_train]
            sweep_centroids[aid] = agent_proj.mean(axis=0)
            sidx += n_train
        # Evaluate per-run accuracy and EER
        sweep_genuine_pca = []
        sweep_impostor_pca = []
        sweep_correct_pca = 0
        sweep_total_pca = 0
        for aid in agent_ids:
            test_texts_pca = response_texts[aid][n_train:]
            for run_idx in range(n_test):
                txt = test_texts_pca[run_idx]
                emb_raw = emb_gen._adapter.embed(txt)
                proj_pca = pca_sweep.transform(emb_raw.reshape(1, -1))[0]
                sweep_total_pca += 1
                best_pca = min(
                    agent_ids,
                    key=lambda x: euclidean_distance(proj_pca, sweep_centroids[x]),
                )
                if best_pca == aid:
                    sweep_correct_pca += 1
                # Genuine distance
                d_own = euclidean_distance(proj_pca, sweep_centroids[aid])
                sweep_genuine_pca.append(d_own)
                # Impostor distances
                for other in agent_ids:
                    if other == aid:
                        continue
                    d_imp = euclidean_distance(proj_pca, sweep_centroids[other])
                    sweep_impostor_pca.append(d_imp)
        pca_eer, _ = _compute_eer(np.array(sweep_genuine_pca), np.array(sweep_impostor_pca))
        pca_acc = sweep_correct_pca / max(sweep_total_pca, 1) * 100
        print(f"  {n_comp_actual:>8d} {pca_eer*100:>7.1f}% {pca_acc:>7.0f}%")

    # ---------------------------------------------------------------
    # Bootstrap Confidence Intervals
    # ---------------------------------------------------------------
    print("\n  Bootstrap Confidence Intervals (20 resamples):")
    eer_samples = []
    for boot in range(20):
        rng = np.random.RandomState(boot)
        boot_genuine = rng.choice(embed_genuine, size=len(embed_genuine), replace=True)
        boot_impostor = rng.choice(embed_impostor, size=len(embed_impostor), replace=True)
        boot_eer, _ = _compute_eer(boot_genuine, boot_impostor)
        eer_samples.append(boot_eer)
    print(f"  Embedding EER: {np.mean(eer_samples)*100:.1f}% +/- {np.std(eer_samples)*100:.1f}%")

    # ---------------------------------------------------------------
    # Embedding ROC Curve
    # ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    y_true_roc = np.concatenate([np.zeros(len(embed_genuine)), np.ones(len(embed_impostor))])
    y_scores_roc = np.concatenate([embed_genuine, embed_impostor])

    # For identity verification: lower distance = more likely genuine
    y_scores_flip = -y_scores_roc
    fpr_id, tpr_id, _ = roc_curve(1 - y_true_roc, y_scores_flip)
    auc_id = sk_auc(fpr_id, tpr_id)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_id, tpr_id, 'b-', lw=2, label=f'Embedding ROC (AUC={auc_id:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Embedding Identity Verification ROC')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / "visualizations" / "benchmark"
    os.makedirs(str(output_dir), exist_ok=True)
    plt.savefig(str(output_dir / "embedding_roc.png"), dpi=150)
    plt.close()
    print(f"\n  Embedding ROC AUC: {auc_id:.3f}")
    print(f"  ROC saved to visualizations/benchmark/embedding_roc.png")

    # Verdict
    print("\n  " + "-" * 50)
    best_eer = min(metric_eer, embed_eer, best_w_eer)
    if best_eer == embed_eer:
        print(f"  BEST: Embedding-only (EER={embed_eer*100:.1f}%)")
    elif best_eer == best_w_eer:
        print(f"  BEST: Weighted fusion w={best_w} (EER={best_w_eer*100:.1f}%)")
    else:
        print(f"  BEST: Metric-only (EER={metric_eer*100:.1f}%)")
    print("  " + "-" * 50)
    print()


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_embedding_validation(use_maas)
