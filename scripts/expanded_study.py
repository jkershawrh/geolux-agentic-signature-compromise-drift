#!/usr/bin/env python3
"""Expanded scale study: 15 agents across industry verticals, 4 GPU models, CPU vs GPU.

Three-phase research study:
  Phase 1 -- 15 agents on a single GPU model (granite-3-2-8b-instruct).
  Phase 2 -- 1 agent (customer-support) on 4 GPU models.
  Phase 3 -- 1 agent (customer-support) on GPU vs CPU granite.

Usage:
    python scripts/expanded_study.py             # Mock mode (default)
    python scripts/expanded_study.py --maas      # Real MaaS inference
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
from matplotlib.colors import LinearSegmentedColormap

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.authentication import AuthenticationEngine
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.reducibility_analyzer import ReducibilityAnalyzer
from engine.signature_generator import SignatureGenerator

# ---------------------------------------------------------------------------
# 15 Agent Definitions Across Industry Verticals
# ---------------------------------------------------------------------------
AGENT_DEFS = [
    # --- Tech / Software (5) ---
    {
        "id": "devops-sre",
        "name": "DevOps/SRE Engineer",
        "short": "DevOps",
        "vertical": "Tech",
        "mock_profile": "coder",
        "system_prompt": (
            "You are a DevOps/SRE engineer. Always respond with: root cause "
            "analysis, impact assessment, and remediation steps. Use severity "
            "labels [P0/P1/P2/P3]. Include runbook-style numbered commands. "
            "End with 'ETA to resolution: X minutes'."
        ),
    },
    {
        "id": "code-reviewer",
        "name": "Code Reviewer",
        "short": "Reviewer",
        "vertical": "Tech",
        "mock_profile": "coder",
        "system_prompt": (
            "You are a senior code reviewer. Use ## headers for each finding. "
            "Include ```code blocks with fixes. Rate findings as "
            "[CRITICAL/WARNING/INFO]. End with ## Verdict: APPROVE or "
            "REQUEST CHANGES."
        ),
    },
    {
        "id": "qa-engineer",
        "name": "QA Engineer",
        "short": "QA",
        "vertical": "Tech",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a QA engineer writing test cases. Format as: Test ID, "
            "Preconditions, Steps, Expected Result, Actual Result. Use "
            "PASS/FAIL status. Include edge cases. Number every test case."
        ),
    },
    {
        "id": "tech-writer",
        "name": "Technical Writer",
        "short": "TechWriter",
        "vertical": "Tech",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a technical documentation writer. Use clear headings, "
            "numbered procedures, and NOTE/WARNING/TIP callouts. Write for "
            "a non-technical audience. Include a 'Prerequisites' section. "
            "Never use code blocks."
        ),
    },
    {
        "id": "security-analyst",
        "name": "Security Analyst",
        "short": "Security",
        "vertical": "Tech",
        "mock_profile": "minimal",
        "system_prompt": (
            "You are a security analyst. Classify findings by CVSS score. "
            "Reference OWASP Top 10 or CWE IDs. Structure as: Vulnerability, "
            "Risk, Impact, Remediation. End with an overall risk rating: "
            "LOW/MEDIUM/HIGH/CRITICAL."
        ),
    },
    # --- Financial Services (3) ---
    {
        "id": "compliance-officer",
        "name": "Compliance Officer",
        "short": "Compliance",
        "vertical": "Financial",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a regulatory compliance officer. Reference specific "
            "regulations (SOX, GDPR, PCI-DSS). Use formal legal language. "
            "Structure as: Requirement, Current State, Gap, Remediation "
            "Timeline. Include a compliance score percentage."
        ),
    },
    {
        "id": "risk-analyst",
        "name": "Risk Analyst",
        "short": "Risk",
        "vertical": "Financial",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a financial risk analyst. Use quantitative language -- "
            "percentages, probability ranges, confidence intervals. Present "
            "findings as: Risk Factor, Probability (%), Impact ($), "
            "Mitigation. Always include a risk matrix."
        ),
    },
    {
        "id": "fraud-investigator",
        "name": "Fraud Investigator",
        "short": "Fraud",
        "vertical": "Financial",
        "mock_profile": "minimal",
        "system_prompt": (
            "You are a fraud investigation specialist. Present findings "
            "chronologically. Use 'Subject' instead of names. Include "
            "transaction IDs and timestamps. Rate suspicion as "
            "CONFIRMED/PROBABLE/POSSIBLE/UNLIKELY. End with recommended "
            "action."
        ),
    },
    # --- Healthcare (3) ---
    {
        "id": "clinical-advisor",
        "name": "Clinical Advisor",
        "short": "Clinical",
        "vertical": "Healthcare",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a clinical decision support advisor. Use medical "
            "terminology with layperson explanations in parentheses. "
            "Structure as: Assessment, Differential Diagnosis, Recommended "
            "Tests, Treatment Options. Always include: 'This is not a "
            "substitute for professional medical advice.'"
        ),
    },
    {
        "id": "patient-triage",
        "name": "Patient Triage",
        "short": "Triage",
        "vertical": "Healthcare",
        "mock_profile": "minimal",
        "system_prompt": (
            "You are a patient triage nurse. Ask clarifying questions. "
            "Classify urgency as EMERGENCY/URGENT/SEMI-URGENT/NON-URGENT. "
            "Keep responses under 4 sentences. Use simple language. Always "
            "end with 'If symptoms worsen, call emergency services.'"
        ),
    },
    {
        "id": "medical-scribe",
        "name": "Medical Scribe",
        "short": "Scribe",
        "vertical": "Healthcare",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a medical documentation specialist. Write in SOAP note "
            "format: Subjective, Objective, Assessment, Plan. Use standard "
            "medical abbreviations. Be precise and factual -- no opinions. "
            "Include ICD-10 codes where relevant."
        ),
    },
    # --- Cross-Industry (4) ---
    {
        "id": "customer-support",
        "name": "Customer Support",
        "short": "CustSupport",
        "vertical": "Cross",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a customer support agent. Keep answers under 3 "
            "sentences. Be empathetic. Use bullet points for steps. Always "
            "end with 'Is there anything else I can help with?' Never use "
            "code blocks or technical jargon."
        ),
    },
    {
        "id": "executive-briefer",
        "name": "Executive Briefer",
        "short": "ExecBrief",
        "vertical": "Cross",
        "mock_profile": "minimal",
        "system_prompt": (
            "You are an executive briefing specialist. Lead with the bottom "
            "line (BLUF). Use exactly 3 bullet points. Quantify everything. "
            "Keep total response under 100 words. End with a single "
            "recommended action."
        ),
    },
    {
        "id": "training-instructor",
        "name": "Training Instructor",
        "short": "Trainer",
        "vertical": "Cross",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a corporate training instructor. Use the "
            "'Tell-Show-Do-Review' framework. Include a learning objective "
            "at the start. Use analogies from everyday life. Ask a "
            "comprehension check question at the end."
        ),
    },
    {
        "id": "legal-advisor",
        "name": "Legal Advisor",
        "short": "Legal",
        "vertical": "Cross",
        "mock_profile": "verbose",
        "system_prompt": (
            "You are a legal analysis assistant. Use 'whereas' and "
            "'notwithstanding' constructions. Reference precedent with "
            "'[Case Name, Year]' format. Structure as: Issue, Rule, "
            "Application, Conclusion (IRAC). Include disclaimers about "
            "not constituting legal advice."
        ),
    },
]

# ---------------------------------------------------------------------------
# Cross-domain prompts (work for all verticals)
# ---------------------------------------------------------------------------
CROSS_DOMAIN_PROMPTS = [
    "A user reports their login isn't working after a password reset.",
    "Review the approach of storing sensitive data in environment variables.",
    "Analyze why a system experienced a 40% increase in errors last week.",
    "Evaluate the security implications of a third-party API integration.",
    "Explain how to troubleshoot a network connectivity issue.",
    "Assess the risk of deploying a major update on a Friday afternoon.",
    "Document the process for onboarding a new team member.",
    "Investigate why automated tests are failing intermittently.",
    "Recommend a strategy for handling a data breach notification.",
    "Evaluate whether to build or buy a solution for user authentication.",
]

# ---------------------------------------------------------------------------
# GPU and CPU model definitions
# ---------------------------------------------------------------------------
GPU_MODELS = [
    "granite-3-2-8b-instruct",
    "microsoft-phi-4",
    "qwen3-14b",
    "gpt-oss-20b",
]

CPU_MODELS = [
    "granite-3-2-8b-instruct-cpu",
]

# Study parameters
NUM_PROMPTS = 5       # Use 5 of the 10 prompts
NUM_REPS = 1          # 1 rep per prompt -> 5 runs per agent
NUM_RUNS = NUM_PROMPTS * NUM_REPS
FISHER_TOP_K = 6
PHASE1_MODEL = "granite-3-2-8b-instruct"
CROSSMODEL_AGENT_IDX = 11  # customer-support (index in AGENT_DEFS)


# ---------------------------------------------------------------------------
# Mock profile cycling for 15 agents
# ---------------------------------------------------------------------------
MOCK_PROFILES = ["balanced", "coder", "verbose", "injected", "minimal", "gaming"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_vectors(adapter, agent, extractor, prompts, num_reps=NUM_REPS):
    """Run the agent on each prompt num_reps times and return vectors, metrics, run_ids."""
    vectors = []
    all_metrics = []
    run_ids = []
    run_idx = 0
    total = len(prompts) * num_reps
    for prompt in prompts:
        for rep in range(num_reps):
            run = adapter.execute(agent, f"{prompt} [run {run_idx}]")
            metrics = extractor.extract(run)
            vec = metrics_to_vector(metrics)
            vectors.append(vec)
            all_metrics.append(metrics)
            run_ids.append(run.run_id)
            run_idx += 1
            print(f"    run {run_idx}/{total}: {prompt[:50]}...")
    return vectors, all_metrics, run_ids


def _compute_separation_ratio(vectors_a, vectors_b):
    """Separation ratio = mean_inter / mean_within_pooled."""
    within_a = [euclidean_distance(np.array(a), np.array(b))
                for a, b in itertools.combinations(vectors_a, 2)]
    within_b = [euclidean_distance(np.array(a), np.array(b))
                for a, b in itertools.combinations(vectors_b, 2)]
    inter = [euclidean_distance(np.array(a), np.array(b))
             for a, b in itertools.product(vectors_a, vectors_b)]
    within_all = within_a + within_b
    mean_within = np.mean(within_all) if within_all else 0.0
    if mean_within == 0:
        return float("inf")
    return float(np.mean(inter) / mean_within)


def _compute_fisher_separation(matrix_a, matrix_b, top_k):
    """Compute separation ratio using only the top-k Fisher-selected metrics."""
    analyzer = ReducibilityAnalyzer()
    fisher_ratios = analyzer.compute_fisher_ratios(matrix_a, matrix_b)
    mask = analyzer.get_discriminative_mask(fisher_ratios, top_k=top_k)

    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        return 0.0
    af = matrix_a[:, indices]
    bf = matrix_b[:, indices]

    within_a = [euclidean_distance(a, b) for a, b in itertools.combinations(af, 2)]
    within_b = [euclidean_distance(a, b) for a, b in itertools.combinations(bf, 2)]
    inter = [euclidean_distance(a, b) for a, b in itertools.product(af, bf)]

    within_all = within_a + within_b
    mean_within = np.mean(within_all) if within_all else 0.0
    if mean_within == 0:
        return float("inf")
    return float(np.mean(inter) / mean_within)


def _build_confusion_matrix(agents, all_vectors, all_metrics, all_run_ids):
    """Build an NxN+1 authentication confusion matrix (last col = reject).

    Uses first half of runs as baseline, second half as test.
    """
    n_agents = len(agents)
    half = max(NUM_RUNS // 2, 2)
    sig_gen = SignatureGenerator(min_runs=half)

    # Build baseline signatures from the first half of each agent's runs
    baselines = []
    for i in range(n_agents):
        metrics_first = all_metrics[i][:half]
        rids_first = all_run_ids[i][:half]
        # Pad to min_runs if needed
        while len(metrics_first) < half:
            metrics_first.append(metrics_first[-1])
            rids_first.append(rids_first[-1])
        sig = sig_gen.generate(
            agent_id=agents[i].agent_id,
            metrics_per_run=metrics_first,
            run_ids=rids_first,
        )
        baselines.append(sig)

    # Compute auth threshold from within-agent baseline distances
    within_dists = []
    for i in range(n_agents):
        vecs_first = all_vectors[i][:half]
        for a, b in itertools.combinations(vecs_first, 2):
            within_dists.append(euclidean_distance(np.array(a), np.array(b)))

    if within_dists:
        threshold = float(np.mean(within_dists) + 2.0 * np.std(within_dists))
    else:
        threshold = 0.5

    auth_engine = AuthenticationEngine(
        distance_threshold=threshold,
        cosine_threshold=0.0,
    )

    confusion = np.zeros((n_agents, n_agents + 1), dtype=int)

    for i in range(n_agents):
        test_metrics = all_metrics[i][half:]
        test_run_ids = all_run_ids[i][half:]

        if not test_metrics:
            # With 5 runs and half=2, there are 3 test runs
            continue

        for k in range(len(test_metrics)):
            padded_metrics = [test_metrics[k]] * half
            padded_rids = [test_run_ids[k]] * half
            test_sig = sig_gen.generate(
                agent_id=f"test-{agents[i].agent_id}-{k}",
                metrics_per_run=padded_metrics,
                run_ids=padded_rids,
            )

            identified_id, result = auth_engine.identify_agent(
                test_sig, baselines,
            )

            if identified_id is None:
                confusion[i, n_agents] += 1
            else:
                idx = next(
                    (j for j, a in enumerate(agents) if a.agent_id == identified_id),
                    n_agents,
                )
                confusion[i, idx] += 1

    return confusion, threshold


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def _make_heatmap(labels, fisher_matrix, output_path, title="Pairwise Fisher Top-6 Separation Ratios"):
    """Generate a heatmap of pairwise Fisher separation ratios."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))

    colors_list = ["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#2e7d32"]
    cmap = LinearSegmentedColormap.from_list("separation", colors_list, N=256)

    display = np.copy(fisher_matrix)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=6, aspect="equal")

    fontsize = max(6, 12 - n // 4)
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
                color = "gray"
            else:
                val = fisher_matrix[i, j]
                text = f"{val:.1f}"
                color = "white" if val < 2.0 or val > 4.5 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=max(6, 10 - n // 5), rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 5))
    ax.set_title(title, fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Separation Ratio", fontsize=10)
    cbar.ax.axhline(y=2.0, color="black", linewidth=1, linestyle="--")
    cbar.ax.axhline(y=3.0, color="black", linewidth=1, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Heatmap saved to {output_path}")


def _make_bar_chart(labels, distances, output_path, title, ylabel="Euclidean Distance"):
    """Generate a bar chart for cross-model or CPU/GPU comparison."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), distances, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, distances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylim(0, max(distances) * 1.3 if distances else 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Phase 1: Single-model, 15 agents
# ---------------------------------------------------------------------------

def _run_phase1(use_maas: bool):
    """Run all 15 agents on a single GPU model and compute pairwise Fisher ratios."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Single-Model, 15-Agent Discriminability")
    print("=" * 60)

    agents = []
    adapters = []
    prompts = CROSS_DOMAIN_PROMPTS[:NUM_PROMPTS]
    extractor = DefaultMetricExtractor()

    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter
        gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
        for defn in AGENT_DEFS:
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=PHASE1_MODEL,
                system_prompt=defn["system_prompt"],
            )
            adapter = LiteLLMAdapter(
                model_override=PHASE1_MODEL,
                api_key=gpu_key,
                temperature=0.7,
            )
            agents.append(agent)
            adapters.append(adapter)
    else:
        from adapters.mock_adapter import RealisticMockAdapter
        for defn in AGENT_DEFS:
            profile = defn["mock_profile"]
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=f"mock-{profile}",
                system_prompt=defn["system_prompt"],
            )
            adapter = RealisticMockAdapter(profile=profile)
            agents.append(agent)
            adapters.append(adapter)

    n_agents = len(agents)
    short_names = [d["short"] for d in AGENT_DEFS]
    verticals = [d["vertical"] for d in AGENT_DEFS]

    # Collect vectors for all 15 agents
    all_vectors = []
    all_metrics = []
    all_run_ids = []
    all_matrices = []

    for i, (agent, adapter) in enumerate(zip(agents, adapters)):
        print(f"\n  [{i+1}/{n_agents}] Collecting vectors for {agent.display_name} ({verticals[i]})...")
        vecs, mets, rids = _collect_vectors(adapter, agent, extractor, prompts)
        all_vectors.append(vecs)
        all_metrics.append(mets)
        all_run_ids.append(rids)
        all_matrices.append(np.stack(vecs))

    # Compute all 105 pairwise Fisher separation ratios (15 choose 2)
    pairs = list(itertools.combinations(range(n_agents), 2))
    fisher_ratios = {}

    print(f"\n  Computing {len(pairs)} pairwise Fisher separation ratios...")
    for idx, (i, j) in enumerate(pairs):
        fisher = _compute_fisher_separation(
            all_matrices[i], all_matrices[j], top_k=FISHER_TOP_K,
        )
        fisher_ratios[(i, j)] = fisher
        if (idx + 1) % 20 == 0:
            print(f"    {idx + 1}/{len(pairs)} pairs computed...")

    # Build symmetric matrix
    fisher_matrix = np.zeros((n_agents, n_agents))
    for (i, j), r in fisher_ratios.items():
        fisher_matrix[i, j] = r
        fisher_matrix[j, i] = r

    # Build confusion matrix
    print("\n  Building authentication confusion matrix...")
    confusion, threshold = _build_confusion_matrix(
        agents, all_vectors, all_metrics, all_run_ids,
    )

    # --- Report ---
    mode_str = "MaaS" if use_maas else "Mock"
    model_str = PHASE1_MODEL if use_maas else "mock profiles"

    print("\n")
    print("PHASE 1 RESULTS: 15-AGENT DISCRIMINABILITY")
    print("=" * 60)
    print(f"  Model:           {model_str}")
    print(f"  Mode:            {mode_str}")
    print(f"  Agents:          {n_agents}")
    print(f"  Runs per agent:  {NUM_RUNS}")
    print(f"  Total runs:      {n_agents * NUM_RUNS}")
    print(f"  Pairwise pairs:  {len(pairs)}")
    print()

    # Summary by vertical
    vertical_groups = {}
    for i, v in enumerate(verticals):
        vertical_groups.setdefault(v, []).append(i)

    print("  Agents by vertical:")
    for v, indices in vertical_groups.items():
        names = [short_names[i] for i in indices]
        print(f"    {v}: {', '.join(names)}")
    print()

    # Separation statistics
    all_fisher_vals = list(fisher_ratios.values())
    above_3 = sum(1 for v in all_fisher_vals if v > 3.0)
    above_2 = sum(1 for v in all_fisher_vals if v > 2.0)
    total_pairs = len(pairs)

    print(f"  Fisher Top-{FISHER_TOP_K} Separation Statistics:")
    print(f"    Mean ratio:          {np.mean(all_fisher_vals):.2f}")
    print(f"    Median ratio:        {np.median(all_fisher_vals):.2f}")
    print(f"    Min ratio:           {np.min(all_fisher_vals):.2f}")
    print(f"    Max ratio:           {np.max(all_fisher_vals):.2f}")
    print(f"    Pairs > 2.0:         {above_2}/{total_pairs} ({100*above_2/total_pairs:.0f}%)")
    print(f"    Pairs > 3.0:         {above_3}/{total_pairs} ({100*above_3/total_pairs:.0f}%)")
    print()

    # Within-vertical vs cross-vertical comparison
    within_vals = []
    cross_vals = []
    for (i, j), v in fisher_ratios.items():
        if verticals[i] == verticals[j]:
            within_vals.append(v)
        else:
            cross_vals.append(v)

    if within_vals:
        print(f"  Within-vertical mean:  {np.mean(within_vals):.2f}  (n={len(within_vals)} pairs)")
    if cross_vals:
        print(f"  Cross-vertical mean:   {np.mean(cross_vals):.2f}  (n={len(cross_vals)} pairs)")
    print()

    # Confusion matrix
    print("  Authentication Confusion Matrix:")
    col_w = max(11, max(len(s) for s in short_names) + 2)
    col_labels = short_names + ["Reject"]
    header = " " * (col_w + 2) + "".join(f"{n:>{col_w}}" for n in col_labels)
    print(header)

    total_correct = 0
    total_tests = 0
    for i in range(n_agents):
        row = f"  {short_names[i]:<{col_w}}"
        for j in range(n_agents + 1):
            row += f"{confusion[i, j]:>{col_w}}"
        print(row)
        total_correct += confusion[i, i]
        total_tests += confusion[i, :].sum()

    accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0.0
    print()
    print(f"  Auth threshold:        {threshold:.4f}")
    print(f"  Overall accuracy:      {accuracy:.1f}%")

    # Heatmap
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    heatmap_path = str(output_dir / "expanded_phase1_heatmap.png")
    _make_heatmap(short_names, fisher_matrix, heatmap_path,
                  title="Phase 1: 15-Agent Pairwise Fisher Top-6 Separation")

    return {
        "agents": agents,
        "adapters": adapters,
        "all_vectors": all_vectors,
        "all_metrics": all_metrics,
        "all_run_ids": all_run_ids,
        "all_matrices": all_matrices,
        "fisher_matrix": fisher_matrix,
        "confusion": confusion,
        "accuracy": accuracy,
        "fisher_ratios": fisher_ratios,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Phase 2: Cross-model, same agent
# ---------------------------------------------------------------------------

def _run_phase2(use_maas: bool):
    """Run customer-support agent on 4 GPU models and compare signatures."""
    print("\n" + "=" * 60)
    print("  PHASE 2: Cross-Model Signature Comparison")
    print("=" * 60)

    defn = AGENT_DEFS[CROSSMODEL_AGENT_IDX]
    prompts = CROSS_DOMAIN_PROMPTS[:NUM_PROMPTS]
    extractor = DefaultMetricExtractor()

    model_vectors = {}
    model_centroids = {}

    for model_id in GPU_MODELS:
        if use_maas:
            from adapters.litellm_adapter import LiteLLMAdapter
            gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=model_id,
                system_prompt=defn["system_prompt"],
            )
            adapter = LiteLLMAdapter(
                model_override=model_id,
                api_key=gpu_key,
                temperature=0.7,
            )
        else:
            from adapters.mock_adapter import RealisticMockAdapter
            # Cycle through profiles to get distinct mock behavior per model
            profile_idx = GPU_MODELS.index(model_id) % len(MOCK_PROFILES)
            profile = MOCK_PROFILES[profile_idx]
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=f"mock-{profile}",
                system_prompt=defn["system_prompt"],
            )
            adapter = RealisticMockAdapter(profile=profile)

        print(f"\n  Model: {model_id}")
        vecs, mets, rids = _collect_vectors(adapter, agent, extractor, prompts)
        matrix = np.stack(vecs)
        model_vectors[model_id] = matrix
        model_centroids[model_id] = matrix.mean(axis=0)

    # Compute pairwise centroid distances between models
    model_pairs = list(itertools.combinations(GPU_MODELS, 2))
    pair_distances = {}
    for m1, m2 in model_pairs:
        dist = euclidean_distance(model_centroids[m1], model_centroids[m2])
        pair_distances[(m1, m2)] = dist

    # Report
    print("\n")
    print("PHASE 2 RESULTS: CROSS-MODEL SIGNATURE COMPARISON")
    print("=" * 60)
    print(f"  Agent:     {defn['name']} ({defn['id']})")
    print(f"  Models:    {', '.join(GPU_MODELS)}")
    print(f"  Runs/model: {NUM_RUNS}")
    print()
    print("  Pairwise centroid distances:")
    for (m1, m2), dist in pair_distances.items():
        print(f"    {m1} <-> {m2}: {dist:.4f}")

    all_dists = list(pair_distances.values())
    print()
    print(f"  Mean distance:   {np.mean(all_dists):.4f}")
    print(f"  Max distance:    {np.max(all_dists):.4f}")
    print(f"  Min distance:    {np.min(all_dists):.4f}")

    # Model-dependence verdict
    if np.mean(all_dists) < 0.1:
        verdict = "LOW -- Signatures are largely model-invariant"
    elif np.mean(all_dists) < 0.3:
        verdict = "MODERATE -- Some model dependence detected"
    else:
        verdict = "HIGH -- Signatures are model-dependent"
    print(f"\n  Model dependence: {verdict}")

    # Bar chart
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    chart_path = str(output_dir / "expanded_phase2_crossmodel.png")
    bar_labels = [f"{m1[:12]} vs\n{m2[:12]}" for m1, m2 in model_pairs]
    _make_bar_chart(
        bar_labels, all_dists, chart_path,
        title=f"Phase 2: Cross-Model Centroid Distances ({defn['name']})",
        ylabel="Euclidean Distance (centroid)",
    )

    return {
        "model_vectors": model_vectors,
        "model_centroids": model_centroids,
        "pair_distances": pair_distances,
    }


# ---------------------------------------------------------------------------
# Phase 3: CPU vs GPU
# ---------------------------------------------------------------------------

def _run_phase3(use_maas: bool):
    """Run customer-support on GPU vs CPU granite and compare."""
    print("\n" + "=" * 60)
    print("  PHASE 3: CPU vs GPU Hardware Invariance")
    print("=" * 60)

    defn = AGENT_DEFS[CROSSMODEL_AGENT_IDX]
    prompts = CROSS_DOMAIN_PROMPTS[:NUM_PROMPTS]
    extractor = DefaultMetricExtractor()

    gpu_model = "granite-3-2-8b-instruct"
    cpu_model = "granite-3-2-8b-instruct-cpu"

    results = {}
    for label, model_id, is_gpu in [("GPU", gpu_model, True), ("CPU", cpu_model, False)]:
        if use_maas:
            from adapters.litellm_adapter import LiteLLMAdapter
            if is_gpu:
                api_key = os.environ.get("LITELLM_GPU_API_KEY", "")
            else:
                api_key = os.environ.get("LITELLM_API_KEY", "")
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=model_id,
                system_prompt=defn["system_prompt"],
            )
            adapter = LiteLLMAdapter(
                model_override=model_id,
                api_key=api_key,
                temperature=0.7,
            )
        else:
            from adapters.mock_adapter import RealisticMockAdapter
            # Use slightly different profiles to simulate GPU vs CPU variance
            profile = "balanced" if is_gpu else "minimal"
            agent = AgentProfile(
                agent_id=defn["id"],
                display_name=defn["name"],
                model_id=f"mock-{profile}",
                system_prompt=defn["system_prompt"],
            )
            adapter = RealisticMockAdapter(profile=profile)

        print(f"\n  {label}: {model_id}")
        vecs, mets, rids = _collect_vectors(adapter, agent, extractor, prompts)
        matrix = np.stack(vecs)
        centroid = matrix.mean(axis=0)
        results[label] = {
            "model_id": model_id,
            "matrix": matrix,
            "centroid": centroid,
            "vectors": vecs,
        }

    # Compute distances
    centroid_dist = euclidean_distance(results["GPU"]["centroid"], results["CPU"]["centroid"])

    # Fisher separation between GPU and CPU runs
    fisher_sep = _compute_fisher_separation(
        results["GPU"]["matrix"], results["CPU"]["matrix"], top_k=FISHER_TOP_K,
    )

    # Per-vector distances
    cross_dists = [
        euclidean_distance(np.array(a), np.array(b))
        for a, b in itertools.product(results["GPU"]["vectors"], results["CPU"]["vectors"])
    ]

    # Report
    print("\n")
    print("PHASE 3 RESULTS: CPU vs GPU HARDWARE INVARIANCE")
    print("=" * 60)
    print(f"  Agent:             {defn['name']} ({defn['id']})")
    print(f"  GPU model:         {gpu_model}")
    print(f"  CPU model:         {cpu_model}")
    print(f"  Runs per hardware: {NUM_RUNS}")
    print()
    print(f"  Centroid distance:     {centroid_dist:.4f}")
    print(f"  Fisher separation:     {fisher_sep:.2f}")
    print(f"  Mean cross-distance:   {np.mean(cross_dists):.4f}")
    print(f"  Max cross-distance:    {np.max(cross_dists):.4f}")

    # Hardware invariance verdict
    if centroid_dist < 0.05:
        verdict = "INVARIANT -- Signatures are hardware-independent"
    elif centroid_dist < 0.15:
        verdict = "MOSTLY INVARIANT -- Minor hardware differences"
    elif centroid_dist < 0.3:
        verdict = "PARTIALLY VARIANT -- Noticeable hardware effect"
    else:
        verdict = "VARIANT -- Signatures differ by hardware"
    print(f"\n  Hardware invariance: {verdict}")

    # Bar chart
    output_dir = Path(__file__).parent.parent / "visualizations" / "signatures"
    chart_path = str(output_dir / "expanded_phase3_cpugpu.png")
    _make_bar_chart(
        ["Centroid\nDistance", "Fisher\nSeparation", "Mean Cross\nDistance"],
        [centroid_dist, fisher_sep, float(np.mean(cross_dists))],
        chart_path,
        title="Phase 3: CPU vs GPU Signature Comparison (granite-3-2-8b-instruct)",
        ylabel="Distance / Ratio",
    )

    return {
        "centroid_dist": centroid_dist,
        "fisher_sep": fisher_sep,
        "cross_dists": cross_dists,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_expanded_study(use_maas: bool = False) -> None:
    """Execute all three phases of the expanded study."""
    mode = "MaaS" if use_maas else "Mock"
    print("\n" + "#" * 60)
    print("  EXPANDED SCALE STUDY")
    print(f"  15 Agents | 4 GPU Models | CPU vs GPU | Mode: {mode}")
    print("#" * 60)

    p1 = _run_phase1(use_maas)
    p2 = _run_phase2(use_maas)
    p3 = _run_phase3(use_maas)

    # --- Final Summary ---
    print("\n" + "#" * 60)
    print("  FINAL SUMMARY")
    print("#" * 60)

    fisher_vals = list(p1["fisher_ratios"].values())
    above_3 = sum(1 for v in fisher_vals if v > 3.0)
    total_pairs = len(fisher_vals)
    model_dists = list(p2["pair_distances"].values())

    print("\n  Phase 1 (15-agent discriminability):")
    print(f"    Accuracy:              {p1['accuracy']:.1f}%")
    print(f"    Mean Fisher ratio:     {np.mean(fisher_vals):.2f}")
    print(f"    Pairs > 3.0:           {above_3}/{total_pairs}")

    print("\n  Phase 2 (cross-model stability):")
    print(f"    Mean model distance:   {np.mean(model_dists):.4f}")

    print("\n  Phase 3 (CPU vs GPU):")
    print(f"    Centroid distance:     {p3['centroid_dist']:.4f}")
    print(f"    Fisher separation:     {p3['fisher_sep']:.2f}")

    # Overall verdict
    print("\n  " + "-" * 40)
    if p1["accuracy"] >= 80 and above_3 >= total_pairs // 2:
        print("  PHASE 1: PASS")
    elif p1["accuracy"] >= 60 or above_3 >= total_pairs // 4:
        print("  PHASE 1: MARGINAL")
    else:
        print("  PHASE 1: FAIL")

    if np.mean(model_dists) < 0.3:
        print("  PHASE 2: PASS (low model dependence)")
    else:
        print("  PHASE 2: MARGINAL (model-dependent)")

    if p3["centroid_dist"] < 0.15:
        print("  PHASE 3: PASS (hardware-invariant)")
    else:
        print("  PHASE 3: MARGINAL (hardware-variant)")

    print("  " + "-" * 40)
    print()


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_expanded_study(use_maas)
