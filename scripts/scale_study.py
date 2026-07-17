#!/usr/bin/env python3
"""Scale study: test 5 agents on the same model to measure pairwise discriminability.

Usage:
    python scripts/scale_study.py             # Mock mode
    python scripts/scale_study.py --maas      # Real MaaS inference
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

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.authentication import AuthenticationEngine
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.reducibility_analyzer import ReducibilityAnalyzer
from engine.signature_generator import SignatureGenerator

# ---------------------------------------------------------------------------
# 5 Agent Definitions
# ---------------------------------------------------------------------------
AGENT_DEFS = [
    {
        "id": "support",
        "name": "Customer Support",
        "short": "Support",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a customer support agent. Keep answers under 3 sentences. "
            "Use simple language. Use bullet points for steps. Always end with "
            "'Is there anything else I can help with?' Never use code blocks or headers."
        ),
    },
    {
        "id": "reviewer",
        "name": "Code Reviewer",
        "short": "Reviewer",
        "mock_profile": "coder",
        "system_prompt": (
            "You are a senior code reviewer. Always use markdown headers "
            "(## Finding, ## Recommendation). Include ```python code blocks. "
            "Number findings. Rate as [CRITICAL]/[WARNING]/[INFO]. "
            "End with ## Verdict: APPROVE or REQUEST CHANGES."
        ),
    },
    {
        "id": "analyst",
        "name": "Data Analyst",
        "short": "Analyst",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a data analyst. Present findings with percentages and statistics. "
            "Use phrases like 'the data suggests', 'statistically significant', "
            "'correlation'. Structure responses as: Key Finding, Supporting Data, "
            "Confidence Level. Always hedge conclusions."
        ),
    },
    {
        "id": "auditor",
        "name": "Security Auditor",
        "short": "Auditor",
        "mock_profile": "minimal",
        "system_prompt": (
            "You are a security auditor. List findings as bullet points with severity "
            "[CRITICAL/HIGH/MEDIUM/LOW]. Include remediation steps for each finding. "
            "Reference OWASP, CVE, or CIS benchmarks. End with a risk score out of 10."
        ),
    },
    {
        "id": "teacher",
        "name": "Teacher",
        "short": "Teacher",
        "mock_profile": "injected",
        "system_prompt": (
            "You are a patient teacher. Use analogies and metaphors like "
            "'think of it like...' Always include a simple example. Ask a follow-up "
            "question to check understanding. Use encouraging phrases like "
            "'Great question!' Avoid jargon."
        ),
    },
]

# ---------------------------------------------------------------------------
# Role-specific prompts that exercise all 5 roles distinctly
# ---------------------------------------------------------------------------
ROLE_PROMPTS = [
    "A user's login isn't working after a password reset.",
    "Review the approach of using a global variable to share state between functions.",
    "Analyze the trend: website traffic increased 40% but revenue only increased 5%.",
    "Evaluate the security of storing API keys in environment variables.",
    "Explain how a for loop works to someone who has never programmed.",
    "The checkout page shows an error for some users but not others.",
    "Assess whether using recursion or iteration is better for traversing a tree.",
    "The conversion rate dropped from 3.2% to 2.1% after a redesign. What happened?",
    "Review the security implications of using HTTP instead of HTTPS for an internal API.",
    "Teach the concept of variables using a real-world analogy.",
]

NUM_PROMPTS = 5
NUM_REPS = 2
NUM_RUNS = NUM_PROMPTS * NUM_REPS  # 10 runs per agent
FISHER_TOP_K = 6
BASE_MODEL = "granite-3-2-8b-instruct-cpu"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _build_agents_and_adapters(use_maas: bool):
    """Return (agents, adapters) — lists of 5 each, aligned by index."""
    agents = []
    adapters = []

    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter

        for defn in AGENT_DEFS:
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=BASE_MODEL,
                system_prompt=defn["system_prompt"],
            )
            adapter = LiteLLMAdapter(model_override=BASE_MODEL, temperature=0.7)
            agents.append(agent)
            adapters.append(adapter)
    else:
        from adapters.mock_adapter import RealisticMockAdapter

        for defn in AGENT_DEFS:
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=f"mock-{defn['mock_profile']}",
                system_prompt=defn["system_prompt"],
            )
            adapter = RealisticMockAdapter(profile=defn["mock_profile"])
            agents.append(agent)
            adapters.append(adapter)

    return agents, adapters


def _collect_vectors(adapter, agent, extractor, prompts, use_maas: bool):
    """Run the agent on each prompt NUM_REPS times and return metric vectors + metrics lists."""
    vectors = []
    all_metrics = []
    run_ids = []
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
            all_metrics.append(metrics)
            run_ids.append(run.run_id)
            run_idx += 1
            print(f"    run {run_idx}/{total}: {prompt[:50]}...")
    return vectors, all_metrics, run_ids


def _pairwise_within(vectors):
    """All pairwise Euclidean distances within one set of vectors."""
    distances = []
    for a, b in itertools.combinations(vectors, 2):
        distances.append(euclidean_distance(np.array(a), np.array(b)))
    return distances


def _pairwise_inter(vectors_a, vectors_b):
    """All pairwise Euclidean distances between two sets."""
    distances = []
    for a, b in itertools.product(vectors_a, vectors_b):
        distances.append(euclidean_distance(np.array(a), np.array(b)))
    return distances


def _compute_separation_ratio(vectors_a, vectors_b):
    """Separation ratio = mean_inter / mean_within_pooled."""
    within_a = _pairwise_within(vectors_a)
    within_b = _pairwise_within(vectors_b)
    inter = _pairwise_inter(vectors_a, vectors_b)
    mean_within = (np.mean(within_a) + np.mean(within_b)) / 2
    if mean_within == 0:
        return float("inf")
    return np.mean(inter) / mean_within


def _compute_fisher_separation(matrix_a, matrix_b, top_k):
    """Compute separation ratio using only the top-k Fisher-selected metrics."""
    analyzer = ReducibilityAnalyzer()
    fisher_ratios = analyzer.compute_fisher_ratios(matrix_a, matrix_b)
    mask = analyzer.get_discriminative_mask(fisher_ratios, top_k=top_k)

    indices = [i for i, m in enumerate(mask) if m]
    af = matrix_a[:, indices]
    bf = matrix_b[:, indices]

    within_a = [euclidean_distance(a, b)
                for a, b in itertools.combinations(af, 2)]
    within_b = [euclidean_distance(a, b)
                for a, b in itertools.combinations(bf, 2)]
    inter = [euclidean_distance(a, b)
             for a, b in itertools.product(af, bf)]

    mean_within = (np.mean(within_a) + np.mean(within_b)) / 2
    if mean_within == 0:
        return float("inf")
    return np.mean(inter) / mean_within


def _build_confusion_matrix(agents, all_vectors, all_metrics, all_run_ids):
    """Build a 5x5 authentication confusion matrix + reject column.

    For each agent, generates a baseline signature from runs 0..4 (first half)
    and tests runs 5..9 (second half) against all baselines.
    """
    n_agents = len(agents)
    half = NUM_RUNS // 2
    sig_gen = SignatureGenerator(min_runs=half)

    # Build baseline signatures from the first half of each agent's runs
    baselines = []
    for i in range(n_agents):
        metrics_first_half = all_metrics[i][:half]
        run_ids_first_half = all_run_ids[i][:half]
        sig = sig_gen.generate(
            agent_id=agents[i].agent_id,
            metrics_per_run=metrics_first_half,
            run_ids=run_ids_first_half,
        )
        baselines.append(sig)

    # Determine auth threshold from within-agent distances of baselines
    within_dists = []
    for i in range(n_agents):
        vecs_first = all_vectors[i][:half]
        for a, b in itertools.combinations(vecs_first, 2):
            within_dists.append(euclidean_distance(np.array(a), np.array(b)))

    if within_dists:
        # Threshold = mean + 2*std of within-agent distances
        threshold = np.mean(within_dists) + 2.0 * np.std(within_dists)
    else:
        threshold = 0.5

    auth_engine = AuthenticationEngine(
        distance_threshold=threshold,
        cosine_threshold=0.0,  # rely on distance only for this study
    )

    # confusion[i][j] = count of agent i's test runs identified as agent j
    # confusion[i][n_agents] = reject count
    confusion = np.zeros((n_agents, n_agents + 1), dtype=int)

    for i in range(n_agents):
        # Test with second half of runs
        test_metrics = all_metrics[i][half:]
        test_run_ids = all_run_ids[i][half:]

        for k in range(len(test_metrics)):
            # Generate a single-run signature for identification
            test_sig = sig_gen.generate(
                agent_id=f"test-{agents[i].agent_id}-{k}",
                metrics_per_run=[test_metrics[k]] * half,  # replicate to meet min_runs
                run_ids=[test_run_ids[k]] * half,
            )

            identified_id, result = auth_engine.identify_agent(
                test_sig, baselines,
            )

            if identified_id is None:
                confusion[i, n_agents] += 1  # reject
            else:
                # Map identified agent_id back to index
                idx = next(
                    (j for j, a in enumerate(agents) if a.agent_id == identified_id),
                    n_agents,
                )
                confusion[i, idx] += 1

    return confusion, threshold


def _make_heatmap(labels, fisher_matrix, output_path):
    """Generate a heatmap of pairwise Fisher separation ratios."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color mapping: green > 3.0, yellow 2.0-3.0, red < 2.0
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#2e7d32"]
    cmap = LinearSegmentedColormap.from_list("separation", colors_list, N=256)

    display = np.copy(fisher_matrix)

    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=6, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
                color = "gray"
            else:
                val = fisher_matrix[i, j]
                text = f"{val:.1f}"
                color = "white" if val < 2.0 or val > 4.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Pairwise Fisher Top-6 Separation Ratios", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Separation Ratio", fontsize=10)

    # Add threshold annotations to colorbar
    cbar.ax.axhline(y=2.0, color="black", linewidth=1, linestyle="--")
    cbar.ax.axhline(y=3.0, color="black", linewidth=1, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_study(use_maas: bool = False) -> None:
    agents, adapters = _build_agents_and_adapters(use_maas)
    extractor = DefaultMetricExtractor()
    prompts = ROLE_PROMPTS[:NUM_PROMPTS]
    n_agents = len(agents)
    short_names = [d["short"] for d in AGENT_DEFS]

    # --- Collect vectors for all 5 agents ---
    all_vectors = []   # list of list[np.ndarray], one per agent
    all_metrics = []   # list of list[list[MetricMeasurement]], one per agent
    all_run_ids = []   # list of list[str], one per agent
    all_matrices = []  # list of np.ndarray (N_RUNS x 29), one per agent

    for i, (agent, adapter) in enumerate(zip(agents, adapters)):
        print(f"\n  Collecting vectors for {agent.display_name}...")
        vecs, mets, rids = _collect_vectors(adapter, agent, extractor, prompts, use_maas)
        all_vectors.append(vecs)
        all_metrics.append(mets)
        all_run_ids.append(rids)
        all_matrices.append(np.stack(vecs))

    # --- Compute all 10 pairwise separation ratios ---
    pairs = list(itertools.combinations(range(n_agents), 2))
    raw_ratios = {}
    fisher_ratios = {}

    for i, j in pairs:
        raw = _compute_separation_ratio(all_vectors[i], all_vectors[j])
        fisher = _compute_fisher_separation(
            all_matrices[i], all_matrices[j], top_k=FISHER_TOP_K,
        )
        raw_ratios[(i, j)] = raw
        fisher_ratios[(i, j)] = fisher

    # Build symmetric matrices for display
    raw_matrix = np.zeros((n_agents, n_agents))
    fisher_matrix = np.zeros((n_agents, n_agents))
    for (i, j), r in raw_ratios.items():
        raw_matrix[i, j] = r
        raw_matrix[j, i] = r
    for (i, j), r in fisher_ratios.items():
        fisher_matrix[i, j] = r
        fisher_matrix[j, i] = r

    # --- Build authentication confusion matrix ---
    print("\n  Building authentication confusion matrix...")
    confusion, threshold = _build_confusion_matrix(
        agents, all_vectors, all_metrics, all_run_ids,
    )

    # --- Report ---
    mode_str = "MaaS" if use_maas else "Mock"
    model_str = BASE_MODEL if use_maas else "mock profiles"

    print("\n")
    print("SCALE STUDY: 5-AGENT DISCRIMINABILITY")
    print("=" * 45)
    print(f"Agents: {', '.join(short_names)}")
    print(f"Model: {model_str}")
    print(f"Mode: {mode_str}")
    print(f"Runs per agent: {NUM_RUNS}")
    print()

    # Pairwise separation table
    col_w = 12
    header = " " * 14 + "".join(f"{n:>{col_w}}" for n in short_names)
    print("Pairwise Separation Ratios (Raw 29-D / Fisher Top-6):")
    print(header)

    for i in range(n_agents):
        row = f"  {short_names[i]:<12}"
        for j in range(n_agents):
            if i == j:
                cell = "-"
            else:
                key = (min(i, j), max(i, j))
                r = raw_ratios[key]
                f = fisher_ratios[key]
                cell = f"{r:.1f}/{f:.1f}"
            row += f"{cell:>{col_w}}"
        print(row)

    # Confusion matrix
    print()
    print("Authentication Confusion Matrix:")
    col_labels = short_names + ["Reject"]
    header = " " * 14 + "".join(f"{n:>{col_w}}" for n in col_labels)
    print(header)

    total_correct = 0
    total_tests = 0
    for i in range(n_agents):
        row = f"  {short_names[i]:<12}"
        for j in range(n_agents + 1):
            row += f"{confusion[i, j]:>{col_w}}"
        print(row)
        total_correct += confusion[i, i]
        total_tests += confusion[i, :].sum()

    accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0.0

    # Summary statistics
    raw_above_3 = sum(1 for v in raw_ratios.values() if v > 3.0)
    fisher_above_3 = sum(1 for v in fisher_ratios.values() if v > 3.0)
    total_pairs = len(pairs)

    print()
    print(f"Auth threshold (distance): {threshold:.4f}")
    print(f"Overall accuracy: {accuracy:.1f}%")
    print(f"Pairs with ratio > 3.0 (raw): {raw_above_3}/{total_pairs}")
    print(f"Pairs with ratio > 3.0 (Fisher): {fisher_above_3}/{total_pairs}")

    # --- Heatmap ---
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    heatmap_path = str(output_dir / "scale_heatmap.png")
    _make_heatmap(short_names, fisher_matrix, heatmap_path)

    # --- Final verdict ---
    print("\n" + "=" * 45)
    print("  FINAL VERDICT")
    print("=" * 45)
    print(f"  Accuracy:                    {accuracy:.1f}%")
    print(f"  Raw pairs > 3.0:            {raw_above_3}/{total_pairs}")
    print(f"  Fisher pairs > 3.0:         {fisher_above_3}/{total_pairs}")
    mean_raw = np.mean(list(raw_ratios.values()))
    mean_fisher = np.mean(list(fisher_ratios.values()))
    print(f"  Mean raw ratio:              {mean_raw:.2f}")
    print(f"  Mean Fisher top-{FISHER_TOP_K} ratio:    {mean_fisher:.2f}")

    if accuracy >= 80 and fisher_above_3 >= total_pairs // 2:
        print("\n  PASS -- Agents are reliably distinguishable at scale")
    elif accuracy >= 60 or fisher_above_3 >= 3:
        print("\n  MARGINAL -- Partial discrimination; needs tuning")
    else:
        print("\n  FAIL -- Insufficient discrimination between agents")


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_study(use_maas)
