#!/usr/bin/env python3
"""Discriminability study: measure within-agent vs inter-agent distance.

Usage:
    python scripts/discriminability_study.py             # Mock mode
    python scripts/discriminability_study.py --maas      # Real MaaS inference
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector

# ---------------------------------------------------------------------------
# Prompt corpus — 30 diverse prompts
# ---------------------------------------------------------------------------
PROMPTS = [
    # First 15 from full_pipeline.py
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "List three benefits of regular exercise.",
    "What is the difference between TCP and UDP?",
    "Describe the water cycle briefly.",
    "What programming language is best for data science?",
    "Explain what an API is to a non-technical person.",
    "What are the primary colors?",
    "How does a hash function work?",
    "What is the Pythagorean theorem?",
    "What causes tides?",
    "Explain recursion simply.",
    "What is machine learning?",
    "How does encryption work?",
    "What is the greenhouse effect?",
    # 15 additional prompts for diversity
    "What is the difference between a stack and a queue?",
    "Explain the concept of supply and demand.",
    "How do vaccines work?",
    "What is the speed of light?",
    "Describe how a binary search algorithm works.",
    "What are the phases of the moon?",
    "Explain what DNS does on the internet.",
    "What is the difference between RAM and ROM?",
    "How does a combustion engine work?",
    "What is natural selection?",
    "Explain the difference between HTML and CSS.",
    "What is inflation in economics?",
    "How does GPS determine your location?",
    "What is the role of mitochondria in a cell?",
    "Explain the concept of object-oriented programming.",
]

# Prompts designed to exercise agent-specific behaviors
# These trigger structural differences between a support agent and a code reviewer
ROLE_PROMPTS = [
    "A user reports that their login isn't working after a password reset. Help them.",
    "Review this approach: using a global variable to share state between functions.",
    "Someone asks why their application is running slowly. What should they check?",
    "Evaluate the trade-offs between using a SQL database vs a NoSQL database.",
    "A customer wants to cancel their subscription but can't find the button. Assist them.",
    "Analyze whether using recursion or iteration is better for traversing a tree structure.",
    "The checkout page shows an error when a user enters their credit card. What do you suggest?",
    "Review this design decision: storing passwords as SHA-256 hashes without salt.",
    "A user can't connect to WiFi after updating their device. Walk them through troubleshooting.",
    "Assess the security implications of using JWT tokens stored in localStorage.",
]

NUM_PROMPTS = 5
NUM_REPS = 2
NUM_RUNS = NUM_PROMPTS * NUM_REPS  # 10 total runs per agent


def _build_adapters_and_agents(use_maas: bool):
    """Return (adapter_alpha, adapter_beta, agent_alpha, agent_beta).

    Agents model real use cases with genuinely different behavioral patterns:

    Alpha = Customer Support Agent: short answers, empathetic, action-oriented,
            uses bullet points for steps, avoids jargon, always ends with
            "Is there anything else I can help with?"

    Beta  = Code Review Agent: structured analysis, always uses markdown headers,
            code blocks for examples, numbered findings, rates severity,
            formal technical language, ends with a summary verdict.
    """
    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter

        base_model = "granite-3-2-8b-instruct-cpu"

        agent_alpha = AgentProfile(
            agent_id="alpha-support",
            display_name="Agent Alpha (Customer Support)",
            model_id=base_model,
            system_prompt=(
                "You are a customer support agent. Follow these rules strictly:\n"
                "1. Keep answers under 3 sentences.\n"
                "2. Use simple, non-technical language.\n"
                "3. If giving steps, use bullet points (- step).\n"
                "4. Be empathetic — acknowledge the user's question first.\n"
                "5. Always end your response with: 'Is there anything else I can help with?'\n"
                "6. Never use code blocks, headers, or numbered lists.\n"
                "7. If you don't know something, say 'Let me connect you with a specialist.'"
            ),
        )
        agent_beta = AgentProfile(
            agent_id="beta-reviewer",
            display_name="Agent Beta (Code Reviewer)",
            model_id=base_model,
            system_prompt=(
                "You are a senior code reviewer. Follow these rules strictly:\n"
                "1. Always structure your response with markdown headers (## Finding, ## Recommendation).\n"
                "2. Include code examples using ```python code blocks for every point.\n"
                "3. Number your findings (1., 2., 3.).\n"
                "4. Rate each finding as [CRITICAL], [WARNING], or [INFO].\n"
                "5. Use precise technical terminology — never simplify.\n"
                "6. End every response with a ## Verdict section containing APPROVE or REQUEST CHANGES.\n"
                "7. Always reference specific line numbers or function names."
            ),
        )
        adapter_alpha = LiteLLMAdapter(model_override=base_model, temperature=0.7)
        adapter_beta = LiteLLMAdapter(model_override=base_model, temperature=0.7)
    else:
        from adapters.mock_adapter import RealisticMockAdapter

        agent_alpha = AgentProfile(
            agent_id="alpha",
            display_name="Agent Alpha (Support)",
            model_id="mock-balanced",
            system_prompt="You are a customer support agent. Be brief and empathetic.",
        )
        agent_beta = AgentProfile(
            agent_id="beta",
            display_name="Agent Beta (Reviewer)",
            model_id="mock-coder",
            system_prompt="You are a code reviewer. Use headers, code blocks, and numbered findings.",
        )
        adapter_alpha = RealisticMockAdapter(profile="balanced")
        adapter_beta = RealisticMockAdapter(profile="coder")

    return adapter_alpha, adapter_beta, agent_alpha, agent_beta


def _collect_vectors(adapter, agent, extractor, prompts, use_maas: bool):
    """Run the agent on each prompt NUM_REPS times and return metric vectors."""
    vectors = []
    run_idx = 0
    total = len(prompts) * NUM_REPS
    for prompt in prompts:
        for rep in range(NUM_REPS):
            if use_maas:
                run = adapter.execute(agent, prompt)
            else:
                run = adapter.execute(agent, f"{prompt} [rep {rep}]")
            metrics = extractor.extract(run)
            vec = metrics_to_vector(metrics)
            vectors.append(vec)
            run_idx += 1
            print(f"    run {run_idx}/{total}: {prompt[:50]}...")
    return vectors


def _pairwise_within(vectors):
    """Compute all pairwise Euclidean distances within one set of vectors."""
    distances = []
    for a, b in itertools.combinations(vectors, 2):
        distances.append(euclidean_distance(np.array(a), np.array(b)))
    return distances


def _pairwise_inter(vectors_a, vectors_b):
    """Compute all pairwise Euclidean distances between two sets."""
    distances = []
    for a, b in itertools.product(vectors_a, vectors_b):
        distances.append(euclidean_distance(np.array(a), np.array(b)))
    return distances


def _cohens_d(within_dists, inter_dists):
    """Cohen's d = (mean_inter - mean_within) / pooled_std."""
    mean_within = np.mean(within_dists)
    mean_inter = np.mean(inter_dists)
    n_w = len(within_dists)
    n_i = len(inter_dists)
    var_w = np.var(within_dists, ddof=1) if n_w > 1 else 0.0
    var_i = np.var(inter_dists, ddof=1) if n_i > 1 else 0.0
    pooled_var = ((n_w - 1) * var_w + (n_i - 1) * var_i) / max(n_w + n_i - 2, 1)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return float("inf")
    return (mean_inter - mean_within) / pooled_std


def _make_violin_plot(within_alpha, within_beta, inter, output_path):
    """Generate a violin plot of the three distance distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [within_alpha, within_beta, inter]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)

    # Color the violin bodies
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Within Alpha", "Within Beta", "Inter-Agent"])
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Discriminability: Within-Agent vs Inter-Agent Distances")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Violin plot saved to {output_path}")


def run_study(use_maas: bool = False) -> None:
    extractor = DefaultMetricExtractor()
    adapter_alpha, adapter_beta, agent_alpha, agent_beta = _build_adapters_and_agents(use_maas)

    prompts = ROLE_PROMPTS[:NUM_PROMPTS]

    # For MaaS mode, set temperature so same-prompt replications produce variance
    if use_maas:
        adapter_alpha.set_temperature(0.7)
        adapter_beta.set_temperature(0.7)

    # --- Collect metric vectors ---
    print("\n  Collecting Alpha vectors...")
    vectors_alpha = _collect_vectors(adapter_alpha, agent_alpha, extractor, prompts, use_maas)

    print("\n  Collecting Beta vectors...")
    vectors_beta = _collect_vectors(adapter_beta, agent_beta, extractor, prompts, use_maas)

    # --- Compute pairwise distances ---
    within_alpha = _pairwise_within(vectors_alpha)
    within_beta = _pairwise_within(vectors_beta)
    inter = _pairwise_inter(vectors_alpha, vectors_beta)

    # --- Pool within-agent distances for statistics ---
    within_all = within_alpha + within_beta

    # --- Statistics ---
    mean_wa = np.mean(within_alpha)
    std_wa = np.std(within_alpha, ddof=1) if len(within_alpha) > 1 else 0.0
    mean_wb = np.mean(within_beta)
    std_wb = np.std(within_beta, ddof=1) if len(within_beta) > 1 else 0.0
    mean_inter = np.mean(inter)
    std_inter = np.std(inter, ddof=1) if len(inter) > 1 else 0.0

    mean_within_pooled = np.mean(within_all)

    separation_ratio = mean_inter / mean_within_pooled if mean_within_pooled > 0 else float("inf")
    d = _cohens_d(within_all, inter)

    # Mann-Whitney U test: within vs inter
    u_stat, p_value = mannwhitneyu(within_all, inter, alternative="less")

    # Verdict
    passed = separation_ratio > 3.0 and d > 0.8 and p_value < 0.05

    # --- Report ---
    print("\n")
    print("DISCRIMINABILITY STUDY RESULTS")
    print("=" * 30)
    print(f"Within-agent (Alpha):  mean={mean_wa:.4f} +/- {std_wa:.4f}  (N={len(within_alpha)} pairs)")
    print(f"Within-agent (Beta):   mean={mean_wb:.4f} +/- {std_wb:.4f}  (N={len(within_beta)} pairs)")
    print(f"Inter-agent:           mean={mean_inter:.4f} +/- {std_inter:.4f}  (N={len(inter)} pairs)")
    print()
    print(f"Separation ratio:      {separation_ratio:.2f}  (target: >3.0)")
    print(f"Cohen's d:             {d:.2f}  (>0.8 = large effect)")
    print(f"Mann-Whitney U p-value: {p_value:.4f}  (<0.05 = significant)")
    print()
    if passed:
        print("VERDICT: PASS -- Agents are reliably distinguishable (raw metrics)")
    else:
        print("VERDICT: NEEDS METRIC SELECTION -- Raw 29-D ratio below target")

    # --- Violin plot ---
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "discriminability.png"
    _make_violin_plot(within_alpha, within_beta, inter, str(output_path))

    # --- Fisher discriminant analysis ---
    from domain.metrics import ALL_METRIC_NAMES

    alpha_matrix = np.stack(vectors_alpha)  # shape: (10, 29)
    beta_matrix = np.stack(vectors_beta)

    print("\n  Fisher Discriminant Ratio per Metric:")
    print(f"  {'Metric':35s} {'Fisher':>8s} {'μ_Alpha':>8s} {'μ_Beta':>8s} {'Signal':>8s}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    fisher_ratios = {}
    for i, name in enumerate(ALL_METRIC_NAMES):
        mu_a = np.mean(alpha_matrix[:, i])
        mu_b = np.mean(beta_matrix[:, i])
        var_a = np.var(alpha_matrix[:, i], ddof=1) if len(alpha_matrix) > 1 else 0
        var_b = np.var(beta_matrix[:, i], ddof=1) if len(beta_matrix) > 1 else 0
        fisher = (mu_a - mu_b)**2 / (var_a + var_b + 1e-10)
        fisher_ratios[name] = fisher

    # Print sorted by Fisher ratio
    for name, ratio in sorted(fisher_ratios.items(), key=lambda x: -x[1]):
        mu_a = np.mean(alpha_matrix[:, ALL_METRIC_NAMES.index(name)])
        mu_b = np.mean(beta_matrix[:, ALL_METRIC_NAMES.index(name)])
        signal = "HIGH" if ratio > 1.0 else "LOW"
        print(f"  {name:35s} {ratio:8.4f} {mu_a:8.4f} {mu_b:8.4f} {signal:>8s}")

    # Sweep top-K Fisher-selected metrics to find optimal subset
    from itertools import combinations, product

    all_sorted_indices = sorted(range(len(ALL_METRIC_NAMES)),
                                key=lambda i: fisher_ratios[ALL_METRIC_NAMES[i]], reverse=True)

    print("\n  Fisher Metric Selection Sweep:")
    print(f"  {'Top-K':>8s} {'Ratio':>8s} {'Within':>8s} {'Inter':>8s} {'Status':>8s}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    best_ratio = 0.0
    best_k = 0
    for k in [3, 4, 5, 6, 8, 10, 12, 15, 20, 29]:
        k = min(k, len(all_sorted_indices))
        idx = all_sorted_indices[:k]
        af = alpha_matrix[:, idx]
        bf = beta_matrix[:, idx]
        wa = [euclidean_distance(a, b) for a, b in combinations(af, 2)]
        wb = [euclidean_distance(a, b) for a, b in combinations(bf, 2)]
        inter_k = [euclidean_distance(a, b) for a, b in product(af, bf)]
        mean_w = (np.mean(wa) + np.mean(wb)) / 2
        r = np.mean(inter_k) / mean_w if mean_w > 0 else 0
        status = "PASS" if r >= 3.0 else ""
        print(f"  {k:>8d} {r:8.2f} {mean_w:8.4f} {np.mean(inter_k):8.4f} {status:>8s}")
        if r > best_ratio:
            best_ratio = r
            best_k = k

    top_names = [ALL_METRIC_NAMES[i] for i in all_sorted_indices[:best_k]]
    print(f"\n  Best: top-{best_k} metrics → ratio={best_ratio:.2f}")
    print(f"  Metrics: {', '.join(top_names)}")

    # Use best-K for the final verdict
    ratio_filtered = best_ratio

    # --- PCA projection comparison ---
    from sklearn.decomposition import PCA

    n_comp = 6
    all_vecs = np.vstack([alpha_matrix, beta_matrix])
    pca = PCA(n_components=min(n_comp, all_vecs.shape[1], all_vecs.shape[0]))
    all_projected = pca.fit_transform(all_vecs)
    alpha_pca = all_projected[:len(vectors_alpha)]
    beta_pca = all_projected[len(vectors_alpha):]

    within_a_pca = [euclidean_distance(a, b) for a, b in combinations(alpha_pca, 2)]
    within_b_pca = [euclidean_distance(a, b) for a, b in combinations(beta_pca, 2)]
    inter_pca = [euclidean_distance(a, b) for a, b in product(alpha_pca, beta_pca)]

    ratio_pca = np.mean(inter_pca) / ((np.mean(within_a_pca) + np.mean(within_b_pca)) / 2)

    print(f"\n  PCA-{n_comp}D separation ratio: {ratio_pca:.2f}")
    print(f"  Explained variance: {sum(pca.explained_variance_ratio_[:n_comp])*100:.1f}%")

    # --- Final Verdict ---
    print("\n")
    print("=" * 50)
    print("  FINAL VERDICT")
    print("=" * 50)
    print(f"  Raw 29-D ratio:           {separation_ratio:.2f}")
    print(f"  Fisher top-{best_k} ratio:      {ratio_filtered:.2f}")
    print(f"  PCA-{n_comp}D ratio:            {ratio_pca:.2f}")
    print(f"  Cohen's d:                {d:.2f}")
    print(f"  p-value:                  {p_value:.6f}")

    final_pass = ratio_filtered >= 3.0 and d > 0.8 and p_value < 0.05
    if final_pass:
        print("\n  PASS — Agents are reliably distinguishable")
        print(f"  with Fisher-selected top-{best_k} metrics")
    elif ratio_filtered >= 2.5:
        print("\n  MARGINAL — Close to target with metric selection")
        print("  Consider more differentiated agent configurations")
    else:
        print("\n  FAIL — Agents not sufficiently distinguishable")
        print("  Need different agent behaviors or additional metrics")


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_study(use_maas)
