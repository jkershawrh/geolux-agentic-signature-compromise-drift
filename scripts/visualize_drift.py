#!/usr/bin/env python3
"""Visualize drift detection results.

Generates:
  - Per-dimension drift bar chart for each perturbation scenario
  - Grouped bar chart comparing all 4 scenarios across 7 dimensions
  - Circular gauge/meter showing drift magnitude per scenario

Usage:
    python scripts/visualize_drift.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from domain.enums import MetricDimension
from domain.geometry import DriftMeasurement
from domain.models import AgentProfile
from engine.baseline_engine import BaselineEngine
from engine.drift_detector import DriftDetector
from engine.run_orchestrator import RunOrchestrator
from engine.signature_generator import SignatureGenerator

plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path(__file__).parent.parent
DRIFT_DIR = ROOT / "visualizations" / "drift"

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

PERTURBATION_PROFILES = {
    "prompt_injection": ("injected", "Prompt Injection"),
    "model_swap": ("minimal", "Model Swap"),
    "temperature_drift": ("verbose", "Temperature Drift"),
    "context_poisoning": ("balanced", "Context Poisoning"),
}


def _severity_color(val: float) -> str:
    """Return color based on drift severity thresholds."""
    if val < 0.1:
        return "#4CAF50"  # green
    elif val < 0.3:
        return "#FFC107"  # yellow/amber
    else:
        return "#F44336"  # red


def _build_drift_data() -> dict[str, DriftMeasurement]:
    """Run the pipeline and compute drift for all 4 perturbation scenarios."""
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")

    agent = AgentProfile(
        agent_id="alpha",
        display_name="Agent Alpha (Balanced)",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant. Answer clearly and concisely.",
    )

    adapter_baseline = RealisticMockAdapter(profile="balanced")
    baseline = BaselineEngine(
        adapter=adapter_baseline, extractor=extractor, generator=generator,
        convergence_epsilon=0.5, convergence_window=2,
    ).establish_baseline(agent, PROMPTS)

    detector = DriftDetector(n_permutations=500)
    results = {}

    for scenario_name, (profile, _label) in PERTURBATION_PROFILES.items():
        perturbed_adapter = RealisticMockAdapter(profile=profile)
        orchestrator = RunOrchestrator(
            adapter=perturbed_adapter, extractor=extractor, generator=generator,
        )
        result = orchestrator.execute_scenario(agent, "healthy_baseline", max_prompts=10)
        if result.signature:
            drift = detector.detect(baseline.signature, result.signature)
            results[scenario_name] = drift

    return results


def plot_dimension_bars(drift: DriftMeasurement, scenario: str, label: str) -> str:
    """Horizontal bar chart of 7-dimension drift values for a scenario."""
    dims = [d.value for d in MetricDimension]
    dim_labels = [d.replace("_", " ").title() for d in dims]
    values = [drift.per_dimension_drift.get(d, 0.0) for d in dims]

    # Sort by value descending for readability
    sorted_pairs = sorted(zip(dim_labels, values, dims), key=lambda x: x[1])
    dim_labels_s, values_s, dims_s = zip(*sorted_pairs)

    colors = [_severity_color(v) for v in values_s]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(dim_labels_s)), values_s, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.6)

    # Value labels
    for bar_obj, val in zip(bars, values_s):
        ax.text(bar_obj.get_width() + 0.005, bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")

    ax.set_yticks(range(len(dim_labels_s)))
    ax.set_yticklabels(dim_labels_s, fontsize=10)
    ax.set_xlabel("Drift Distance", fontsize=11)
    ax.set_title(f"Per-Dimension Drift: {label}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values_s) * 1.25 + 0.01)

    # Threshold reference lines
    ax.axvline(0.1, color="#FFC107", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.3, color="#F44336", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.1, len(dim_labels_s) - 0.3, " 0.1", fontsize=7, color="#FFC107")
    ax.text(0.3, len(dim_labels_s) - 0.3, " 0.3", fontsize=7, color="#F44336")

    # Legend
    patches = [
        mpatches.Patch(color="#4CAF50", label="Low (< 0.1)"),
        mpatches.Patch(color="#FFC107", label="Medium (0.1-0.3)"),
        mpatches.Patch(color="#F44336", label="High (> 0.3)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    path = str(DRIFT_DIR / f"dimensions_{scenario}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_comparison(all_drifts: dict[str, DriftMeasurement]) -> str:
    """Grouped bar chart comparing all 4 scenarios across 7 dimensions."""
    dims = [d.value for d in MetricDimension]
    dim_labels = [d.replace("_", " ").title() for d in dims]
    scenarios = list(all_drifts.keys())
    scenario_labels = [PERTURBATION_PROFILES[s][1] for s in scenarios]

    n_dims = len(dims)
    n_scenarios = len(scenarios)
    x = np.arange(n_dims)
    width = 0.8 / n_scenarios

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
        drift = all_drifts[scenario]
        values = [drift.per_dimension_drift.get(d, 0.0) for d in dims]
        offset = (i - n_scenarios / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label,
               color=colors[i % len(colors)], edgecolor="white",
               linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Drift Distance", fontsize=11)
    ax.set_title("Drift Comparison Across Perturbation Scenarios",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # Threshold lines
    ax.axhline(0.1, color="#FFC107", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.3, color="#F44336", linestyle="--", linewidth=1, alpha=0.5)

    ax.grid(axis="y", alpha=0.3)

    path = str(DRIFT_DIR / "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_gauge(drift: DriftMeasurement, scenario: str, label: str) -> str:
    """Circular gauge/meter showing drift magnitude 0-1 with colored zones."""
    magnitude = drift.drift_magnitude

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"aspect": "equal"})

    # Draw the gauge arc (180 degrees, bottom half hidden)
    # Green zone: 0 to 0.1 -> 180 to 162 degrees
    # Yellow zone: 0.1 to 0.3 -> 162 to 126 degrees
    # Red zone: 0.3 to 1.0 -> 126 to 0 degrees
    zone_specs = [
        (0.0, 0.1, "#4CAF50"),   # green
        (0.1, 0.3, "#FFC107"),   # yellow
        (0.3, 1.0, "#F44336"),   # red
    ]

    for lo, hi, color in zone_specs:
        theta_start = 180 - hi * 180
        theta_end = 180 - lo * 180
        wedge = mpatches.Wedge(
            center=(0, 0), r=1.0,
            theta1=theta_start, theta2=theta_end,
            facecolor=color, edgecolor="white", linewidth=1, alpha=0.3,
            width=0.35,
        )
        ax.add_patch(wedge)

    # Filled arc up to the magnitude value
    fill_color = _severity_color(magnitude)
    if magnitude > 0.001:
        fill_wedge = mpatches.Wedge(
            center=(0, 0), r=1.0,
            theta1=180 - magnitude * 180, theta2=180,
            facecolor=fill_color, edgecolor="none", alpha=0.7,
            width=0.35,
        )
        ax.add_patch(fill_wedge)

    # Needle
    angle_rad = np.pi * (1 - magnitude)
    needle_x = 0.85 * np.cos(angle_rad)
    needle_y = 0.85 * np.sin(angle_rad)
    ax.plot([0, needle_x], [0, needle_y], color="#333", linewidth=2.5, zorder=5)
    ax.plot(0, 0, "o", color="#333", markersize=8, zorder=6)

    # Labels
    ax.text(0, -0.15, f"{magnitude:.3f}", fontsize=22, fontweight="bold",
            ha="center", va="center", color=fill_color)
    ax.text(0, -0.35, "Drift Magnitude", fontsize=10, ha="center", va="center",
            color="#666")
    ax.text(0, 1.15, label, fontsize=12, fontweight="bold", ha="center", va="center")

    # Scale labels
    ax.text(-1.08, -0.02, "0", fontsize=8, ha="center", color="#666")
    ax.text(1.08, -0.02, "1", fontsize=8, ha="center", color="#666")
    ax.text(0, 1.02, "0.5", fontsize=7, ha="center", color="#666")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.5, 1.3)
    ax.axis("off")

    path = str(DRIFT_DIR / f"magnitude_{scenario}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(DRIFT_DIR, exist_ok=True)

    print("Building baseline and running perturbation scenarios...")
    all_drifts = _build_drift_data()

    print("Generating per-dimension drift bar charts...")
    for scenario, drift in all_drifts.items():
        label = PERTURBATION_PROFILES[scenario][1]
        path = plot_dimension_bars(drift, scenario, label)
        print(f"  {path}")

    print("Generating scenario comparison chart...")
    comp_path = plot_comparison(all_drifts)
    print(f"  {comp_path}")

    print("Generating drift magnitude gauges...")
    for scenario, drift in all_drifts.items():
        label = PERTURBATION_PROFILES[scenario][1]
        path = plot_gauge(drift, scenario, label)
        print(f"  {path}")

    print("\nDone. All drift visualizations generated.")


if __name__ == "__main__":
    main()
