#!/usr/bin/env python3
"""Correlation study: identify redundant metrics and measure effective dimensionality.

Usage:
    python scripts/correlation_study.py             # Mock mode
    python scripts/correlation_study.py --maas      # Real MaaS inference
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from adapters.metric_extractor import DefaultMetricExtractor
from domain.metrics import ALL_METRIC_NAMES
from domain.models import AgentProfile
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

NUM_RUNS = 30


def _build_adapter_and_agent(use_maas: bool):
    """Return (adapter, agent) for a single agent."""
    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter

        base_model = "granite-3-2-8b-instruct-cpu"
        agent = AgentProfile(
            agent_id="study-agent",
            display_name="Study Agent (Balanced)",
            model_id=base_model,
            system_prompt="You are a helpful assistant. Answer clearly and concisely in 1-3 sentences.",
        )
        adapter = LiteLLMAdapter(model_override=base_model)
    else:
        from adapters.mock_adapter import RealisticMockAdapter

        agent = AgentProfile(
            agent_id="study-agent",
            display_name="Study Agent (Balanced)",
            model_id="mock-balanced",
            system_prompt="You are a helpful assistant. Answer clearly and concisely.",
        )
        adapter = RealisticMockAdapter(profile="balanced")

    return adapter, agent


def _collect_vectors(adapter, agent, extractor, prompts, num_runs):
    """Run the agent num_runs times on rotating prompts and return metric vectors."""
    vectors = []
    for run_idx in range(num_runs):
        prompt = prompts[run_idx % len(prompts)]
        run = adapter.execute(agent, prompt)
        metrics = extractor.extract(run)
        vec = metrics_to_vector(metrics)
        vectors.append(vec)
        print(f"    run {run_idx + 1}/{num_runs}: {prompt[:50]}...")
    return vectors


def _make_heatmap(corr_matrix, metric_names, output_path):
    """Generate a correlation heatmap."""
    n = len(metric_names)
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.matshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_names, rotation=90, fontsize=6)
    ax.set_yticklabels(metric_names, fontsize=6)

    ax.set_title("Metric Correlation Matrix", pad=60, fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Heatmap saved to {output_path}")


def run_study(use_maas: bool = False) -> None:
    extractor = DefaultMetricExtractor()
    adapter, agent = _build_adapter_and_agent(use_maas)

    # --- Collect metric vectors ---
    print("\n  Collecting metric vectors...")
    vectors = _collect_vectors(adapter, agent, extractor, PROMPTS, NUM_RUNS)

    # Build matrix: rows = runs, columns = metrics
    matrix = np.stack(vectors)  # shape (NUM_RUNS, 29)
    n_metrics = matrix.shape[1]
    metric_names = ALL_METRIC_NAMES

    # --- Correlation matrix ---
    # np.corrcoef expects rows=variables, so transpose
    # Handle constant columns (zero variance) that produce NaN correlations
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    # Replace NaN with 0 (happens for constant metrics like sentiment_stability)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # --- Identify redundant pairs ---
    redundant_pairs = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            rho = corr_matrix[i, j]
            if abs(rho) > 0.8:
                redundant_pairs.append((metric_names[i], metric_names[j], rho))

    # Sort by absolute correlation descending
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # --- PCA effective dimensionality ---
    # Standardize columns before PCA (zero-variance columns become zero)
    col_std = matrix.std(axis=0)
    col_std[col_std == 0] = 1.0
    matrix_standardized = (matrix - matrix.mean(axis=0)) / col_std

    pca = PCA()
    pca.fit(matrix_standardized)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = int(np.searchsorted(cumulative_variance, 0.95)) + 1

    # --- Report ---
    print("\n")
    print("CORRELATION STUDY RESULTS")
    print("=" * 25)
    print(f"Total metrics:              {n_metrics}")
    print(f"Highly correlated pairs:    {len(redundant_pairs)}  (|rho| > 0.8)")
    for name_a, name_b, rho in redundant_pairs:
        print(f"  - {name_a} <-> {name_b}    rho={rho:.2f}")

    print()
    print(f"PCA effective dimensionality: {effective_dim}  (components for 95% variance)")
    for i in range(min(effective_dim + 2, n_metrics)):
        pct = pca.explained_variance_ratio_[i] * 100
        print(f"  Component {i + 1}: {pct:.1f}% variance")

    # Use effective_dim from PCA as the independence measure
    print()
    print(f"VERDICT: {effective_dim}/{n_metrics} metrics are effectively independent")

    # --- Heatmap ---
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "correlation_heatmap.png"
    _make_heatmap(corr_matrix, metric_names, str(output_path))


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_study(use_maas)
