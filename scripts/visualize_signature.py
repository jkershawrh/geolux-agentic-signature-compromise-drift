#!/usr/bin/env python3
"""Visualize agent geometric signatures.

Generates:
  - Radar chart comparing two agents across 7 metric dimensions
  - Embedding heatmap of the 29-metric vectors for two agents
  - PCA manifold projection scatter of all run vectors

Usage:
    python scripts/visualize_signature.py
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

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from domain.enums import MetricDimension
from domain.metrics import ALL_METRIC_NAMES, METRIC_DEFINITIONS
from domain.models import AgentProfile
from engine.baseline_engine import BaselineEngine
from engine.geometric.embedding import metrics_to_vector
from engine.signature_generator import SignatureGenerator

plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path(__file__).parent.parent
SIG_DIR = ROOT / "visualizations" / "signatures"
MAN_DIR = ROOT / "visualizations" / "manifolds"

PROMPTS = [
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
]


def _build_baselines() -> tuple:
    """Run mock pipelines for two agents and return baseline results."""
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")

    agent_alpha = AgentProfile(
        agent_id="alpha",
        display_name="Agent Alpha (Balanced)",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant. Answer clearly and concisely.",
    )
    agent_beta = AgentProfile(
        agent_id="beta",
        display_name="Agent Beta (Coder)",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a technical coding assistant. Use code examples.",
    )

    adapter_alpha = RealisticMockAdapter(profile="balanced")
    adapter_beta = RealisticMockAdapter(profile="coder")

    baseline_alpha = BaselineEngine(
        adapter=adapter_alpha, extractor=extractor, generator=generator,
        convergence_epsilon=0.5, convergence_window=2,
    ).establish_baseline(agent_alpha, PROMPTS)

    baseline_beta = BaselineEngine(
        adapter=adapter_beta, extractor=extractor, generator=generator,
        convergence_epsilon=0.5, convergence_window=2,
    ).establish_baseline(agent_beta, PROMPTS)

    return agent_alpha, agent_beta, baseline_alpha, baseline_beta


def _dimension_means(metric_snapshot: dict[str, float]) -> dict[str, float]:
    """Compute the mean normalized value per dimension from a metric snapshot."""
    means = {}
    for dim in MetricDimension:
        dim_metrics = METRIC_DEFINITIONS[dim]
        vals = [metric_snapshot.get(m, 0.0) for m in dim_metrics]
        means[dim.value] = float(np.mean(vals))
    return means


def plot_radar(
    alpha_snapshot: dict[str, float],
    beta_snapshot: dict[str, float],
    alpha_id: str,
    beta_id: str,
) -> list[str]:
    """Radar chart comparing two agents across 7 dimensions."""
    alpha_means = _dimension_means(alpha_snapshot)
    beta_means = _dimension_means(beta_snapshot)

    dims = [d.value for d in MetricDimension]
    labels = [d.replace("_", " ").title() for d in dims]

    alpha_vals = [alpha_means[d] for d in dims]
    beta_vals = [beta_means[d] for d in dims]

    # Close the polygon
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    alpha_vals += alpha_vals[:1]
    beta_vals += beta_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    ax.plot(angles, alpha_vals, "o-", linewidth=2, label=f"Alpha ({alpha_id})",
            color="#2196F3", markersize=6)
    ax.fill(angles, alpha_vals, alpha=0.15, color="#2196F3")

    ax.plot(angles, beta_vals, "s-", linewidth=2, label=f"Beta ({beta_id})",
            color="#FF5722", markersize=6)
    ax.fill(angles, beta_vals, alpha=0.15, color="#FF5722")

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.set_title("Agent Signature Geometry\n(7 Metric Dimensions)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

    paths = []
    for aid in [alpha_id, beta_id]:
        path = str(SIG_DIR / f"radar_{aid}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_heatmap(
    alpha_snapshot: dict[str, float],
    beta_snapshot: dict[str, float],
    alpha_id: str,
    beta_id: str,
) -> list[str]:
    """Embedding heatmap of the 29-metric vector for two agents."""
    alpha_vec = [alpha_snapshot.get(m, 0.0) for m in ALL_METRIC_NAMES]
    beta_vec = [beta_snapshot.get(m, 0.0) for m in ALL_METRIC_NAMES]
    data = np.array([alpha_vec, beta_vec])

    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Alpha ({alpha_id})", f"Beta ({beta_id})"], fontsize=10)
    ax.set_xticks(range(len(ALL_METRIC_NAMES)))
    short_labels = [n.replace("_", "\n") for n in ALL_METRIC_NAMES]
    ax.set_xticklabels(short_labels, fontsize=5.5, rotation=90, ha="center")

    # Dimension separators
    offset = 0
    for dim in MetricDimension:
        count = len(METRIC_DEFINITIONS[dim])
        if offset > 0:
            ax.axvline(offset - 0.5, color="white", linewidth=2)
        offset += count

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Normalized Value", fontsize=9)

    ax.set_title("29-Metric Embedding Vectors (Signature Fingerprint)", fontsize=13,
                 fontweight="bold")

    paths = []
    for aid in [alpha_id, beta_id]:
        path = str(SIG_DIR / f"heatmap_{aid}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_manifold(
    baseline_alpha, baseline_beta,
    alpha_id: str, beta_id: str,
) -> str:
    """PCA 2D scatter of all run vectors for both agents."""
    from sklearn.decomposition import PCA

    alpha_vecs = np.array([metrics_to_vector(m) for m in baseline_alpha.all_metrics])
    beta_vecs = np.array([metrics_to_vector(m) for m in baseline_beta.all_metrics])

    all_vecs = np.vstack([alpha_vecs, beta_vecs])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_vecs)

    n_alpha = len(alpha_vecs)
    proj_alpha = projected[:n_alpha]
    proj_beta = projected[n_alpha:]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(proj_alpha[:, 0], proj_alpha[:, 1], c="#2196F3", s=80,
               edgecolors="white", linewidth=0.8, label=f"Alpha ({alpha_id})",
               alpha=0.85, zorder=3)
    ax.scatter(proj_beta[:, 0], proj_beta[:, 1], c="#FF5722", s=80,
               edgecolors="white", linewidth=0.8, label=f"Beta ({beta_id})",
               alpha=0.85, zorder=3)

    # Centroids
    ca = proj_alpha.mean(axis=0)
    cb = proj_beta.mean(axis=0)
    ax.scatter(*ca, c="#2196F3", s=200, marker="*", edgecolors="black",
               linewidth=1.2, zorder=5)
    ax.scatter(*cb, c="#FF5722", s=200, marker="*", edgecolors="black",
               linewidth=1.2, zorder=5)

    # Connecting line between centroids
    ax.plot([ca[0], cb[0]], [ca[1], cb[1]], "--", color="grey", linewidth=1, alpha=0.6)
    mid = (ca + cb) / 2
    dist = np.linalg.norm(ca - cb)
    ax.annotate(f"d={dist:.3f}", xy=mid, fontsize=9, color="grey", ha="center",
                va="bottom")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=11)
    ax.set_title("Manifold Projection (PCA)\nAgent Behavior Clusters in Reduced Space",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    path = str(MAN_DIR / "projection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(SIG_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)

    print("Building baselines for two agents...")
    agent_alpha, agent_beta, baseline_alpha, baseline_beta = _build_baselines()

    alpha_snap = baseline_alpha.signature.metric_snapshot
    beta_snap = baseline_beta.signature.metric_snapshot

    print("Generating radar charts...")
    radar_paths = plot_radar(alpha_snap, beta_snap, agent_alpha.agent_id,
                             agent_beta.agent_id)
    for p in radar_paths:
        print(f"  {p}")

    print("Generating embedding heatmaps...")
    heatmap_paths = plot_heatmap(alpha_snap, beta_snap, agent_alpha.agent_id,
                                 agent_beta.agent_id)
    for p in heatmap_paths:
        print(f"  {p}")

    print("Generating manifold projection...")
    proj_path = plot_manifold(baseline_alpha, baseline_beta,
                              agent_alpha.agent_id, agent_beta.agent_id)
    print(f"  {proj_path}")

    print("\nDone. All signature visualizations generated.")


if __name__ == "__main__":
    main()
