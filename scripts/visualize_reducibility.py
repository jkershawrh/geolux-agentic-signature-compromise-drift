#!/usr/bin/env python3
"""Visualize computational reducibility analysis.

Generates:
  - Reducibility spectrum: horizontal bar chart of all 29 metrics sorted by
    predictability score, color-coded by reducibility class.

Usage:
    python scripts/visualize_reducibility.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from domain.enums import Reducibility
from domain.models import AgentProfile
from engine.baseline_engine import BaselineEngine
from engine.reducibility_analyzer import ReducibilityAnalyzer
from engine.signature_generator import SignatureGenerator

plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path(__file__).parent.parent
SIG_DIR = ROOT / "visualizations" / "signatures"

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

REDUCIBILITY_COLORS = {
    Reducibility.REDUCIBLE: "#4CAF50",               # green
    Reducibility.CONDITIONALLY_REDUCIBLE: "#FFC107",  # yellow/amber
    Reducibility.IRREDUCIBLE: "#F44336",              # red
}


def _build_reducibility_data():
    """Run baseline and compute reducibility classifications."""
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")

    agent = AgentProfile(
        agent_id="alpha",
        display_name="Agent Alpha (Balanced)",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant. Answer clearly and concisely.",
    )

    adapter = RealisticMockAdapter(profile="balanced")
    baseline = BaselineEngine(
        adapter=adapter, extractor=extractor, generator=generator,
        convergence_epsilon=0.5, convergence_window=2,
    ).establish_baseline(agent, PROMPTS)

    analyzer = ReducibilityAnalyzer(min_samples=5)
    classifications = analyzer.analyze(baseline.all_metrics, agent.agent_id)
    return classifications


def plot_reducibility_spectrum(classifications) -> str:
    """Horizontal bar chart of all 29 metrics sorted by predictability score."""
    # Sort by predictability (high to low)
    sorted_cls = sorted(classifications, key=lambda c: c.predictability_score, reverse=True)

    names = [c.metric_name.replace("_", " ").title() for c in sorted_cls]
    scores = [c.predictability_score for c in sorted_cls]
    variances = [c.variance for c in sorted_cls]
    colors = [REDUCIBILITY_COLORS[c.reducibility] for c in sorted_cls]
    dim_labels = [c.dimension.value.replace("_", " ").title() for c in sorted_cls]

    n = len(sorted_cls)
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.35)))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, scores, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7, alpha=0.85)

    # Variance annotation on each bar
    for i, (bar_obj, var, score, dim) in enumerate(zip(bars, variances, scores, dim_labels)):
        # Value text at end of bar
        text_x = bar_obj.get_width() + 0.01
        ax.text(text_x, bar_obj.get_y() + bar_obj.get_height() / 2,
                f"var={var:.5f}", va="center", fontsize=7, color="#555",
                family="monospace")
        # Dimension label inside bar (if wide enough)
        if score > 0.3:
            ax.text(bar_obj.get_width() - 0.02,
                    bar_obj.get_y() + bar_obj.get_height() / 2,
                    dim, va="center", ha="right", fontsize=6, color="white",
                    fontweight="bold", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()  # highest predictability at top
    ax.set_xlabel("Predictability Score", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_title("Computational Reducibility Spectrum\n(29 Metrics Sorted by Predictability)",
                 fontsize=14, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#4CAF50", label="Reducible (stable core)"),
        mpatches.Patch(color="#FFC107", label="Conditionally Reducible"),
        mpatches.Patch(color="#F44336", label="Irreducible (noise)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    ax.grid(axis="x", alpha=0.3)

    path = str(SIG_DIR / "reducibility.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(SIG_DIR, exist_ok=True)

    print("Running reducibility analysis...")
    classifications = _build_reducibility_data()

    # Print summary
    from engine.reducibility_analyzer import ReducibilityAnalyzer
    summary = ReducibilityAnalyzer().summary(classifications)
    print(f"  Reducible:              {summary['reducible']}")
    print(f"  Conditionally reducible: {summary['conditionally_reducible']}")
    print(f"  Irreducible:            {summary['irreducible']}")

    print("Generating reducibility spectrum...")
    path = plot_reducibility_spectrum(classifications)
    print(f"  {path}")

    print("\nDone. Reducibility visualization generated.")


if __name__ == "__main__":
    main()
