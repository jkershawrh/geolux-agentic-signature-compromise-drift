#!/usr/bin/env python3
"""Identity Validation Suite — 5 experiments to validate behavioral fingerprinting.

Usage:
    python scripts/identity_validation.py             # Mock mode
    python scripts/identity_validation.py --maas      # Real MaaS (GPU)
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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import AgentProfile
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.reducibility_analyzer import ReducibilityAnalyzer


# ---------------------------------------------------------------------------
# 15 Agent Definitions Across Industry Verticals (from expanded_study.py)
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
# 4 Hard-Pair Agents — nearly identical prompts
# ---------------------------------------------------------------------------
HARD_PAIRS = [
    {
        "id": "support-a",
        "name": "Support Agent A",
        "short": "SupportA",
        "vertical": "HardPair",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a customer support agent. Keep answers under 3 sentences. "
            "Use simple language. Use bullet points for steps. Always end with "
            "'Is there anything else I can help with?' Never use code blocks "
            "or technical jargon."
        ),
    },
    {
        "id": "support-b",
        "name": "Support Agent B",
        "short": "SupportB",
        "vertical": "HardPair",
        "mock_profile": "balanced",
        "system_prompt": (
            "You are a customer support agent. Keep answers under 3 sentences. "
            "Use simple language. Use bullet points for steps. Always end with "
            "'How else can I assist you today?' Never use code blocks "
            "or technical jargon."
        ),
    },
    {
        "id": "reviewer-a",
        "name": "Code Reviewer A",
        "short": "ReviewerA",
        "vertical": "HardPair",
        "mock_profile": "coder",
        "system_prompt": (
            "You are a code reviewer. Use ## headers for findings. Include "
            "code blocks. Rate findings as [CRITICAL/WARNING/INFO]. End with "
            "## Verdict: APPROVE or REQUEST CHANGES."
        ),
    },
    {
        "id": "reviewer-b",
        "name": "Code Reviewer B",
        "short": "ReviewerB",
        "vertical": "HardPair",
        "mock_profile": "coder",
        "system_prompt": (
            "You are a code reviewer. Use ## headers for findings. Include "
            "code blocks. Rate findings as [HIGH/MEDIUM/LOW]. End with "
            "## Verdict: PASS or FAIL."
        ),
    },
]

ALL_AGENTS = AGENT_DEFS + HARD_PAIRS

# ---------------------------------------------------------------------------
# 20 Fixed Prompts (same as ASC-Bench)
# ---------------------------------------------------------------------------
PROMPTS = [
    "A user reports their login isn't working after a password reset.",
    "Review the approach of storing sensitive data in environment variables.",
    "Analyze why error rates increased 40% last week.",
    "Evaluate the security implications of a third-party API integration.",
    "Explain how to troubleshoot a network connectivity issue.",
    "Assess the risk of deploying a major update on Friday afternoon.",
    "Document the process for onboarding a new team member.",
    "Investigate why automated tests are failing intermittently.",
    "Recommend a strategy for handling a data breach notification.",
    "Evaluate whether to build or buy a solution for authentication.",
    "A customer wants to cancel their subscription but can't find the button.",
    "Review whether using a microservice architecture is appropriate here.",
    "Analyze the trade-off between consistency and availability in this system.",
    "Evaluate the compliance implications of storing data in a foreign region.",
    "Explain the concept of eventual consistency to a non-technical stakeholder.",
    "Assess the risk of using an open-source dependency with no maintainer.",
    "Document the incident response procedure for a production outage.",
    "Investigate why the recommendation engine is showing irrelevant results.",
    "Recommend an approach for implementing role-based access control.",
    "Evaluate the ethical implications of using AI for automated hiring decisions.",
]

# Study parameters
FISHER_TOP_K = 6
PHASE1_MODEL = "granite-3-2-8b-instruct"
VIZ_DIR = Path(__file__).parent.parent / "visualizations" / "validation"


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def _collect_agent_data(adapter, agent, extractor, prompts):
    """Run agent on prompts, return array of metric vectors (n_prompts x n_metrics)."""
    vectors = []
    for idx, prompt in enumerate(prompts):
        run = adapter.execute(agent, prompt)
        vec = metrics_to_vector(extractor.extract(run))
        vectors.append(vec)
    return np.array(vectors)


def _build_adapter_and_agent(defn, use_maas):
    """Build (adapter, agent) from an agent definition dict."""
    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter
        gpu_key = os.environ.get("LITELLM_GPU_API_KEY", "")
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
    else:
        from adapters.mock_adapter import RealisticMockAdapter
        profile = defn["mock_profile"]
        agent = AgentProfile(
            agent_id=defn["id"],
            display_name=defn["name"],
            model_id=f"mock-{profile}",
            system_prompt=defn["system_prompt"],
        )
        adapter = RealisticMockAdapter(profile=profile)
    return adapter, agent


# ---------------------------------------------------------------------------
# Experiment 1: Scale Test (15 agents, 105 pairwise comparisons)
# ---------------------------------------------------------------------------

def _experiment_1_scale(agent_data, agent_ids, short_names):
    """15 agents, all 105 pairwise Fisher separation ratios.

    Uses first 10 runs as fingerprint, last 10 as verification.
    Returns results dict.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Scale Test (15 agents)")
    print("=" * 60)

    n_agents = len(agent_ids)
    half = agent_data[agent_ids[0]].shape[0] // 2
    if half < 2:
        half = 2

    analyzer = ReducibilityAnalyzer()
    pairs = list(itertools.combinations(range(n_agents), 2))

    # Compute all pairwise Fisher separation ratios
    pairwise_ratios = {}
    fisher_matrix = np.zeros((n_agents, n_agents))

    print(f"  Computing {len(pairs)} pairwise Fisher separation ratios...")
    for idx, (i, j) in enumerate(pairs):
        aid_i = agent_ids[i]
        aid_j = agent_ids[j]
        mat_i = agent_data[aid_i]
        mat_j = agent_data[aid_j]
        fisher_ratios = analyzer.compute_fisher_ratios(mat_i, mat_j)
        mask = analyzer.get_discriminative_mask(fisher_ratios, top_k=FISHER_TOP_K)
        fisher_indices = [k for k, m in enumerate(mask) if m]
        if not fisher_indices:
            ratio = 0.0
        else:
            af = mat_i[:, fisher_indices]
            bf = mat_j[:, fisher_indices]
            within_a = [euclidean_distance(a, b) for a, b in itertools.combinations(af, 2)]
            within_b = [euclidean_distance(a, b) for a, b in itertools.combinations(bf, 2)]
            inter = [euclidean_distance(a, b) for a, b in itertools.product(af, bf)]
            within_all = within_a + within_b
            mean_within = np.mean(within_all) if within_all else 0.0
            ratio = float(np.mean(inter) / mean_within) if mean_within > 0 else float("inf")
        pairwise_ratios[(aid_i, aid_j)] = ratio
        fisher_matrix[i, j] = ratio
        fisher_matrix[j, i] = ratio

    # Build fingerprint centroids from first half, verify with second half
    # Use Fisher-filtered vectors (aggregated top-6 across all pairs)
    all_fisher_ratios_agg = {}
    from domain.metrics import ALL_METRIC_NAMES
    for name in ALL_METRIC_NAMES:
        all_fisher_ratios_agg[name] = 0.0
    for (ai, aj) in pairs:
        aid_i = agent_ids[ai]
        aid_j = agent_ids[aj]
        ratios = analyzer.compute_fisher_ratios(agent_data[aid_i], agent_data[aid_j])
        for m, v in ratios.items():
            all_fisher_ratios_agg[m] += v / len(pairs)
    sorted_metrics = sorted(all_fisher_ratios_agg.items(), key=lambda x: -x[1])
    top_names = {name for name, _ in sorted_metrics[:FISHER_TOP_K]}
    global_fisher_indices = [i for i, name in enumerate(ALL_METRIC_NAMES) if name in top_names]

    # Fingerprint centroids from first half
    fingerprint_centroids = {}
    for aid in agent_ids:
        mat = agent_data[aid][:half, :][:, global_fisher_indices]
        fingerprint_centroids[aid] = mat.mean(axis=0)

    # Batch verification: centroid of second half vs all fingerprints
    batch_correct = 0
    for aid in agent_ids:
        test_mat = agent_data[aid][half:, :][:, global_fisher_indices]
        test_centroid = test_mat.mean(axis=0)
        best_id = None
        best_dist = float("inf")
        for cand_id, fp_centroid in fingerprint_centroids.items():
            d = euclidean_distance(test_centroid, fp_centroid)
            if d < best_dist:
                best_dist = d
                best_id = cand_id
        if best_id == aid:
            batch_correct += 1
    batch_accuracy = batch_correct / n_agents * 100

    # Per-run verification: each individual test run vs all fingerprints
    run_correct = 0
    run_total = 0
    for aid in agent_ids:
        test_mat = agent_data[aid][half:, :][:, global_fisher_indices]
        for vec in test_mat:
            best_id = None
            best_dist = float("inf")
            for cand_id, fp_centroid in fingerprint_centroids.items():
                d = euclidean_distance(vec, fp_centroid)
                if d < best_dist:
                    best_dist = d
                    best_id = cand_id
            if best_id == aid:
                run_correct += 1
            run_total += 1
    per_run_accuracy = run_correct / run_total * 100 if run_total > 0 else 0.0

    # Statistics
    all_ratios = list(pairwise_ratios.values())
    pairs_above_2 = sum(1 for v in all_ratios if v > 2.0)
    pairs_above_3 = sum(1 for v in all_ratios if v > 3.0)

    # Find worst and best pair
    worst_pair_key = min(pairwise_ratios, key=pairwise_ratios.get)
    best_pair_key = max(pairwise_ratios, key=pairwise_ratios.get)

    # Report
    print(f"\n  Pairwise Fisher Top-{FISHER_TOP_K} Separation Statistics:")
    print(f"    Mean ratio:          {np.mean(all_ratios):.2f}")
    print(f"    Median ratio:        {np.median(all_ratios):.2f}")
    print(f"    Min ratio:           {np.min(all_ratios):.2f}  ({worst_pair_key[0]} vs {worst_pair_key[1]})")
    print(f"    Max ratio:           {np.max(all_ratios):.2f}  ({best_pair_key[0]} vs {best_pair_key[1]})")
    print(f"    Pairs > 2.0:         {pairs_above_2}/{len(pairs)} ({100*pairs_above_2/len(pairs):.0f}%)")
    print(f"    Pairs > 3.0:         {pairs_above_3}/{len(pairs)} ({100*pairs_above_3/len(pairs):.0f}%)")
    print()
    print(f"  Verification Accuracy:")
    print(f"    Batch (centroid):    {batch_accuracy:.1f}%")
    print(f"    Per-run:             {per_run_accuracy:.1f}%")

    return {
        "pairwise_ratios": pairwise_ratios,
        "fisher_matrix": fisher_matrix,
        "batch_accuracy": batch_accuracy,
        "per_run_accuracy": per_run_accuracy,
        "worst_pair": worst_pair_key,
        "best_pair": best_pair_key,
        "pairs_above_2": pairs_above_2,
        "pairs_above_3": pairs_above_3,
        "global_fisher_indices": global_fisher_indices,
        "fingerprint_centroids": fingerprint_centroids,
        "short_names": short_names,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Within-Vertical Hard Pairs
# ---------------------------------------------------------------------------

def _experiment_2_hard_pairs(hard_pair_data):
    """Test Support-A vs Support-B and Reviewer-A vs Reviewer-B.

    Reveals the system's discrimination boundary on nearly identical agents.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Hard Pair Discrimination")
    print("=" * 60)

    analyzer = ReducibilityAnalyzer()
    from domain.metrics import ALL_METRIC_NAMES

    pair_results = {}
    hard_pair_defs = [
        ("support-a", "support-b", "Support A vs B"),
        ("reviewer-a", "reviewer-b", "Reviewer A vs B"),
    ]

    for aid_a, aid_b, label in hard_pair_defs:
        mat_a = hard_pair_data[aid_a]
        mat_b = hard_pair_data[aid_b]
        half = mat_a.shape[0] // 2
        if half < 2:
            half = 2

        fisher_ratios = analyzer.compute_fisher_ratios(mat_a, mat_b)
        mask = analyzer.get_discriminative_mask(fisher_ratios, top_k=FISHER_TOP_K)
        fisher_indices = [i for i, m in enumerate(mask) if m]

        # Discriminating metrics (Fisher > 1.0)
        discriminating = [(name, ratio) for name, ratio in fisher_ratios.items() if ratio > 1.0]
        discriminating.sort(key=lambda x: -x[1])

        # Compute separation ratio on Fisher-filtered vectors
        if fisher_indices:
            af = mat_a[:, fisher_indices]
            bf = mat_b[:, fisher_indices]
            within_a = [euclidean_distance(a, b) for a, b in itertools.combinations(af, 2)]
            within_b = [euclidean_distance(a, b) for a, b in itertools.combinations(bf, 2)]
            inter = [euclidean_distance(a, b) for a, b in itertools.product(af, bf)]
            within_all = within_a + within_b
            mean_within = np.mean(within_all) if within_all else 0.0
            ratio = float(np.mean(inter) / mean_within) if mean_within > 0 else float("inf")
        else:
            ratio = 0.0

        # Batch verification
        fp_a = mat_a[:half, :][:, fisher_indices].mean(axis=0) if fisher_indices else np.array([])
        fp_b = mat_b[:half, :][:, fisher_indices].mean(axis=0) if fisher_indices else np.array([])
        batch_correct = 0
        for aid, mat in [(aid_a, mat_a), (aid_b, mat_b)]:
            if not fisher_indices:
                continue
            test_centroid = mat[half:, :][:, fisher_indices].mean(axis=0)
            d_a = euclidean_distance(test_centroid, fp_a)
            d_b = euclidean_distance(test_centroid, fp_b)
            expected_fp = fp_a if aid == aid_a else fp_b
            if aid == aid_a and d_a < d_b:
                batch_correct += 1
            elif aid == aid_b and d_b < d_a:
                batch_correct += 1
        batch_accuracy = batch_correct / 2 * 100

        # Per-run verification
        run_correct = 0
        run_total = 0
        for aid, mat in [(aid_a, mat_a), (aid_b, mat_b)]:
            if not fisher_indices:
                continue
            test_vecs = mat[half:, :][:, fisher_indices]
            for vec in test_vecs:
                d_a = euclidean_distance(vec, fp_a)
                d_b = euclidean_distance(vec, fp_b)
                if aid == aid_a and d_a < d_b:
                    run_correct += 1
                elif aid == aid_b and d_b < d_a:
                    run_correct += 1
                run_total += 1
        per_run_accuracy = run_correct / run_total * 100 if run_total > 0 else 0.0

        print(f"\n  {label}:")
        print(f"    Fisher separation ratio: {ratio:.2f}")
        print(f"    Discriminating metrics (Fisher > 1.0): {len(discriminating)}")
        if discriminating:
            for name, r in discriminating[:5]:
                print(f"      {name:35s}  {r:.4f}")
        print(f"    Batch accuracy:   {batch_accuracy:.1f}%")
        print(f"    Per-run accuracy: {per_run_accuracy:.1f}%")

        pair_results[label] = {
            "ratio": ratio,
            "discriminating_metrics": discriminating,
            "batch_accuracy": batch_accuracy,
            "per_run_accuracy": per_run_accuracy,
        }

    return pair_results


# ---------------------------------------------------------------------------
# Experiment 3: Minimum-Run Sweep
# ---------------------------------------------------------------------------

def _experiment_3_min_runs(agent_data, best_pair, worst_pair, global_fisher_indices):
    """Sweep fingerprint size: 2, 3, 5, 7, 10 runs.

    For each N, bootstrap 20 trials selecting N runs for fingerprint
    and remaining for verification. Reports mean accuracy +/- std.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Minimum-Run Sweep")
    print("=" * 60)

    sweep_ns = [2, 3, 5, 7, 10]
    n_trials = 20

    results = {}
    for pair_label, (aid_a, aid_b) in [("Best (easiest)", best_pair), ("Worst (hardest)", worst_pair)]:
        mat_a = agent_data[aid_a][:, global_fisher_indices]
        mat_b = agent_data[aid_b][:, global_fisher_indices]
        n_runs = min(mat_a.shape[0], mat_b.shape[0])

        print(f"\n  {pair_label} pair: {aid_a} vs {aid_b} ({n_runs} runs each)")
        print(f"  {'N':>5s}  {'Mean Acc':>10s}  {'Std':>8s}")
        print(f"  {'-----':>5s}  {'----------':>10s}  {'--------':>8s}")

        sweep_results = []
        for n_fp in sweep_ns:
            if n_fp >= n_runs:
                sweep_results.append((n_fp, float("nan"), float("nan")))
                print(f"  {n_fp:>5d}  {'N/A':>10s}  {'N/A':>8s}  (not enough runs)")
                continue

            trial_accs = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial)
                idx_a = rng.permutation(n_runs)
                idx_b = rng.permutation(n_runs)

                fp_a = mat_a[idx_a[:n_fp]].mean(axis=0)
                fp_b = mat_b[idx_b[:n_fp]].mean(axis=0)

                test_a = mat_a[idx_a[n_fp:]]
                test_b = mat_b[idx_b[n_fp:]]

                correct = 0
                total = 0
                for vec in test_a:
                    d_a = euclidean_distance(vec, fp_a)
                    d_b = euclidean_distance(vec, fp_b)
                    if d_a < d_b:
                        correct += 1
                    total += 1
                for vec in test_b:
                    d_a = euclidean_distance(vec, fp_a)
                    d_b = euclidean_distance(vec, fp_b)
                    if d_b < d_a:
                        correct += 1
                    total += 1
                trial_accs.append(correct / total * 100 if total > 0 else 0.0)

            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            sweep_results.append((n_fp, mean_acc, std_acc))
            print(f"  {n_fp:>5d}  {mean_acc:>9.1f}%  {std_acc:>7.1f}%")

        results[pair_label] = sweep_results

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Cross-Session Stability
# ---------------------------------------------------------------------------

def _experiment_4_cross_session(adapter_builder, agents_subset_defs, use_maas):
    """Fingerprint with prompts 1-10, verify with prompts 11-20.

    Compares same-session accuracy (split prompts[0:10] into 5/5)
    to cross-session accuracy (fingerprint from prompts[0:10], verify from prompts[10:20]).
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Cross-Session Stability")
    print("=" * 60)

    extractor = DefaultMetricExtractor()
    analyzer = ReducibilityAnalyzer()
    session_a_prompts = PROMPTS[:10]
    session_b_prompts = PROMPTS[10:20]

    # Collect data for both sessions
    session_a_data = {}
    session_b_data = {}
    agent_ids = []

    for defn in agents_subset_defs:
        adapter, agent = adapter_builder(defn, use_maas)
        aid = defn["id"]
        agent_ids.append(aid)
        print(f"  Collecting session A data for {defn['name']}...")
        session_a_data[aid] = _collect_agent_data(adapter, agent, extractor, session_a_prompts)
        print(f"  Collecting session B data for {defn['name']}...")
        session_b_data[aid] = _collect_agent_data(adapter, agent, extractor, session_b_prompts)

    # Compute global Fisher indices from session A
    from domain.metrics import ALL_METRIC_NAMES
    agg_fisher = {name: 0.0 for name in ALL_METRIC_NAMES}
    pairs = list(itertools.combinations(agent_ids, 2))
    for ai, aj in pairs:
        ratios = analyzer.compute_fisher_ratios(session_a_data[ai], session_a_data[aj])
        for m, v in ratios.items():
            agg_fisher[m] += v / len(pairs)
    sorted_m = sorted(agg_fisher.items(), key=lambda x: -x[1])
    top_names = {name for name, _ in sorted_m[:FISHER_TOP_K]}
    fisher_idx = [i for i, name in enumerate(ALL_METRIC_NAMES) if name in top_names]

    # Same-session accuracy: split session A into 5/5
    same_fp = {}
    for aid in agent_ids:
        same_fp[aid] = session_a_data[aid][:5, :][:, fisher_idx].mean(axis=0)
    same_correct = 0
    same_total = 0
    for aid in agent_ids:
        test_vecs = session_a_data[aid][5:, :][:, fisher_idx]
        for vec in test_vecs:
            best_id = min(agent_ids, key=lambda x: euclidean_distance(vec, same_fp[x]))
            if best_id == aid:
                same_correct += 1
            same_total += 1
    same_session_acc = same_correct / same_total * 100 if same_total > 0 else 0.0

    # Cross-session accuracy: fingerprint from session A, verify from session B
    cross_fp = {}
    for aid in agent_ids:
        cross_fp[aid] = session_a_data[aid][:, fisher_idx].mean(axis=0)
    cross_correct = 0
    cross_total = 0
    for aid in agent_ids:
        test_vecs = session_b_data[aid][:, fisher_idx]
        for vec in test_vecs:
            best_id = min(agent_ids, key=lambda x: euclidean_distance(vec, cross_fp[x]))
            if best_id == aid:
                cross_correct += 1
            cross_total += 1
    cross_session_acc = cross_correct / cross_total * 100 if cross_total > 0 else 0.0

    print(f"\n  Same-session accuracy (5/5 split):  {same_session_acc:.1f}%")
    print(f"  Cross-session accuracy (A->B):      {cross_session_acc:.1f}%")
    delta = same_session_acc - cross_session_acc
    print(f"  Stability delta:                    {delta:+.1f}%")

    if abs(delta) < 10:
        verdict = "STABLE -- Cross-session fingerprints are reliable"
    elif abs(delta) < 20:
        verdict = "MODERATE DRIFT -- Some session-dependence detected"
    else:
        verdict = "UNSTABLE -- Fingerprints vary significantly across sessions"
    print(f"  Verdict: {verdict}")

    return {
        "same_session_accuracy": same_session_acc,
        "cross_session_accuracy": cross_session_acc,
        "delta": delta,
        "agent_ids": agent_ids,
    }


# ---------------------------------------------------------------------------
# Experiment 5: False Acceptance Rate (FAR/FRR/EER)
# ---------------------------------------------------------------------------

def _experiment_5_far(agent_data, agent_ids, global_fisher_indices):
    """Sweep threshold to compute FAR/FRR and find EER.

    For each pair (A, B):
      - Genuine distances: A-test to A-fingerprint
      - Impostor distances: B-test to A-fingerprint
    Aggregates across all pairs and sweeps threshold.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: False Acceptance Rate Analysis")
    print("=" * 60)

    half = agent_data[agent_ids[0]].shape[0] // 2
    if half < 2:
        half = 2

    # Build fingerprint centroids
    fingerprint_centroids = {}
    for aid in agent_ids:
        mat = agent_data[aid][:half, :][:, global_fisher_indices]
        fingerprint_centroids[aid] = mat.mean(axis=0)

    # Collect genuine and impostor distances
    genuine_dists = []
    impostor_dists = []

    for aid in agent_ids:
        test_vecs = agent_data[aid][half:, :][:, global_fisher_indices]
        fp = fingerprint_centroids[aid]
        for vec in test_vecs:
            genuine_dists.append(euclidean_distance(vec, fp))

        # Impostor: other agents' test runs against this agent's fingerprint
        for other_aid in agent_ids:
            if other_aid == aid:
                continue
            other_test = agent_data[other_aid][half:, :][:, global_fisher_indices]
            for vec in other_test:
                impostor_dists.append(euclidean_distance(vec, fp))

    genuine_dists = np.array(genuine_dists)
    impostor_dists = np.array(impostor_dists)

    # Sweep thresholds
    max_dist = max(genuine_dists.max(), impostor_dists.max()) if len(genuine_dists) > 0 and len(impostor_dists) > 0 else 1.0
    thresholds = np.linspace(0, max_dist * 1.2, 200)

    far_curve = []
    frr_curve = []
    for t in thresholds:
        far = float(np.mean(impostor_dists < t))  # impostors accepted
        frr = float(np.mean(genuine_dists > t))    # genuine rejected
        far_curve.append(far)
        frr_curve.append(frr)

    far_curve = np.array(far_curve)
    frr_curve = np.array(frr_curve)

    # Find EER (where FAR ~= FRR)
    diff = np.abs(far_curve - frr_curve)
    eer_idx = np.argmin(diff)
    eer = float((far_curve[eer_idx] + frr_curve[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    print(f"\n  Genuine distances:   mean={np.mean(genuine_dists):.4f}, std={np.std(genuine_dists):.4f}")
    print(f"  Impostor distances:  mean={np.mean(impostor_dists):.4f}, std={np.std(impostor_dists):.4f}")
    print(f"  Equal Error Rate:    {eer*100:.2f}%")
    print(f"  EER threshold:       {eer_threshold:.4f}")

    # Report at fixed operating points
    for target_far in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(far_curve - target_far))
        print(f"  At FAR={target_far*100:.0f}%: FRR={frr_curve[idx]*100:.1f}%, threshold={thresholds[idx]:.4f}")

    return {
        "genuine_dists": genuine_dists,
        "impostor_dists": impostor_dists,
        "thresholds": thresholds,
        "far_curve": far_curve,
        "frr_curve": frr_curve,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _plot_scale_heatmap(exp1_results):
    """Plot 1: 15x15 Fisher ratio heatmap."""
    n = len(exp1_results["short_names"])
    labels = exp1_results["short_names"]
    fisher_matrix = exp1_results["fisher_matrix"]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))

    colors_list = ["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#2e7d32"]
    cmap = LinearSegmentedColormap.from_list("separation", colors_list, N=256)

    im = ax.imshow(fisher_matrix, cmap=cmap, vmin=0, vmax=6, aspect="equal")

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
    ax.set_title("Experiment 1: 15-Agent Pairwise Fisher Top-6 Separation", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Separation Ratio", fontsize=10)
    cbar.ax.axhline(y=2.0, color="black", linewidth=1, linestyle="--")
    cbar.ax.axhline(y=3.0, color="black", linewidth=1, linestyle="--")

    plt.tight_layout()
    path = str(VIZ_DIR / "scale_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved to {path}")


def _plot_hard_pairs(exp2_results):
    """Plot 2: Bar chart comparing hard pair metrics."""
    labels = list(exp2_results.keys())
    ratios = [exp2_results[l]["ratio"] for l in labels]
    batch_accs = [exp2_results[l]["batch_accuracy"] for l in labels]
    per_run_accs = [exp2_results[l]["per_run_accuracy"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: separation ratios
    bars = axes[0].bar(labels, ratios, color=["#4C72B0", "#55A868"], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ratios):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Fisher Separation Ratio", fontsize=11)
    axes[0].set_title("Hard Pair Separation", fontsize=12)
    axes[0].axhline(y=2.0, color="#d32f2f", linestyle="--", alpha=0.7, label="Target: 2.0")
    axes[0].legend(fontsize=9)

    # Right: accuracy
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width / 2, batch_accs, width, label="Batch", color="#4C72B0", edgecolor="black")
    axes[1].bar(x + width / 2, per_run_accs, width, label="Per-run", color="#55A868", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("Hard Pair Verification Accuracy", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 110)

    plt.suptitle("Experiment 2: Hard Pair Discrimination", fontsize=14)
    plt.tight_layout()
    path = str(VIZ_DIR / "hard_pairs.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Hard pairs chart saved to {path}")


def _plot_min_runs_curve(exp3_results):
    """Plot 3: Accuracy vs N with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Best (easiest)": "#2e7d32", "Worst (hardest)": "#d32f2f"}
    for pair_label, sweep_data in exp3_results.items():
        ns = []
        means = []
        stds = []
        for n_fp, mean_acc, std_acc in sweep_data:
            if not np.isnan(mean_acc):
                ns.append(n_fp)
                means.append(mean_acc)
                stds.append(std_acc)
        if ns:
            color = colors.get(pair_label, "#333333")
            ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                        label=pair_label, color=color)

    ax.set_xlabel("Fingerprint Size (N runs)", fontsize=12)
    ax.set_ylabel("Verification Accuracy (%)", fontsize=12)
    ax.set_title("Experiment 3: Minimum-Run Sweep", fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color="#999999", linestyle="--", alpha=0.5, label="90% target")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = str(VIZ_DIR / "min_runs_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Min-runs curve saved to {path}")


def _plot_cross_session(exp4_results):
    """Plot 4: Same-session vs cross-session comparison bars."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Same-Session\n(5/5 split)", "Cross-Session\n(A -> B)"]
    values = [exp4_results["same_session_accuracy"], exp4_results["cross_session_accuracy"]]
    colors = ["#4C72B0", "#C44E52"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Identification Accuracy (%)", fontsize=12)
    ax.set_title("Experiment 4: Cross-Session Stability", fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(y=80, color="#999999", linestyle="--", alpha=0.5, label="80% target")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = str(VIZ_DIR / "cross_session.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Cross-session chart saved to {path}")


def _plot_det_curve(exp5_results):
    """Plot 5: DET curve (FAR vs FRR with EER marked)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    far = exp5_results["far_curve"]
    frr = exp5_results["frr_curve"]
    eer = exp5_results["eer"]

    ax.plot(far * 100, frr * 100, color="#2e7d32", lw=2, label="DET Curve")
    ax.plot([0, 100], [0, 100], color="#bdbdbd", lw=1, linestyle="--", label="FAR = FRR")

    # Mark EER
    eer_idx = np.argmin(np.abs(far - frr))
    ax.scatter([far[eer_idx] * 100], [frr[eer_idx] * 100], color="#d32f2f", s=80,
               zorder=5, label=f"EER = {eer*100:.2f}%")

    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("Experiment 5: Detection Error Tradeoff (DET) Curve", fontsize=13)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = str(VIZ_DIR / "det_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  DET curve saved to {path}")


# ---------------------------------------------------------------------------
# Main Flow
# ---------------------------------------------------------------------------

def run_validation(use_maas: bool = False) -> None:
    """Execute all 5 identity validation experiments."""
    mode = "MaaS" if use_maas else "Mock"
    print("\n" + "#" * 60)
    print("  IDENTITY VALIDATION SUITE")
    print(f"  19 Agents | 20 Prompts | 5 Experiments | Mode: {mode}")
    print("#" * 60)

    os.makedirs(str(VIZ_DIR), exist_ok=True)

    embedding_adapter = None
    if use_maas:
        try:
            from adapters.embedding_adapter import EmbeddingAdapter
            embedding_adapter = EmbeddingAdapter(
                api_key=os.environ.get("LITELLM_GPU_API_KEY", ""),
            )
            print("  Embedding adapter: nomic-embed-text-v1-5 (GPU)")
        except Exception as e:
            print(f"  Embedding adapter: unavailable ({e})")

    extractor = DefaultMetricExtractor(embedding_adapter=embedding_adapter)

    # ---------------------------------------------------------------
    # Phase 0: Collect data for all 19 agents on all 20 prompts
    # ---------------------------------------------------------------
    print("\n--- DATA COLLECTION ---")

    # Collect data for the 15 base agents
    base_agent_data = {}
    base_agent_ids = []
    base_short_names = []
    for i, defn in enumerate(AGENT_DEFS):
        adapter, agent = _build_adapter_and_agent(defn, use_maas)
        aid = defn["id"]
        base_agent_ids.append(aid)
        base_short_names.append(defn["short"])
        print(f"  [{i+1}/15] Collecting {defn['name']}...")
        base_agent_data[aid] = _collect_agent_data(adapter, agent, extractor, PROMPTS)

    # Collect data for the 4 hard-pair agents
    hard_pair_data = {}
    for i, defn in enumerate(HARD_PAIRS):
        adapter, agent = _build_adapter_and_agent(defn, use_maas)
        aid = defn["id"]
        print(f"  [HP {i+1}/4] Collecting {defn['name']}...")
        hard_pair_data[aid] = _collect_agent_data(adapter, agent, extractor, PROMPTS)

    # Combined data for experiments that use all 15 base agents
    all_data = {**base_agent_data, **hard_pair_data}

    print(f"\n  Total runs collected: {(len(AGENT_DEFS) + len(HARD_PAIRS)) * len(PROMPTS)}")

    # ---------------------------------------------------------------
    # Experiment 1: Scale Test (15 agents)
    # ---------------------------------------------------------------
    exp1 = _experiment_1_scale(base_agent_data, base_agent_ids, base_short_names)

    # ---------------------------------------------------------------
    # Experiment 2: Hard Pair Discrimination
    # ---------------------------------------------------------------
    exp2 = _experiment_2_hard_pairs(hard_pair_data)

    # ---------------------------------------------------------------
    # Experiment 3: Minimum-Run Sweep
    # ---------------------------------------------------------------
    exp3 = _experiment_3_min_runs(
        base_agent_data,
        exp1["best_pair"],
        exp1["worst_pair"],
        exp1["global_fisher_indices"],
    )

    # ---------------------------------------------------------------
    # Experiment 4: Cross-Session Stability
    # ---------------------------------------------------------------
    # Use 5 agents from different verticals for efficiency
    cross_session_defs = [
        AGENT_DEFS[0],   # DevOps (Tech)
        AGENT_DEFS[5],   # Compliance (Financial)
        AGENT_DEFS[8],   # Clinical (Healthcare)
        AGENT_DEFS[11],  # Customer Support (Cross)
        AGENT_DEFS[14],  # Legal (Cross)
    ]
    exp4 = _experiment_4_cross_session(_build_adapter_and_agent, cross_session_defs, use_maas)

    # ---------------------------------------------------------------
    # Experiment 5: False Acceptance Rate
    # ---------------------------------------------------------------
    exp5 = _experiment_5_far(base_agent_data, base_agent_ids, exp1["global_fisher_indices"])

    # ---------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------
    print("\n--- GENERATING VISUALIZATIONS ---")
    _plot_scale_heatmap(exp1)
    _plot_hard_pairs(exp2)
    _plot_min_runs_curve(exp3)
    _plot_cross_session(exp4)
    _plot_det_curve(exp5)

    # ---------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------
    print("\n" + "#" * 60)
    print("  IDENTITY VALIDATION SUITE — FINAL SUMMARY")
    print("#" * 60)

    all_ratios = list(exp1["pairwise_ratios"].values())
    print(f"\n  Experiment 1 (Scale Test, 15 agents):")
    print(f"    Mean Fisher ratio:     {np.mean(all_ratios):.2f}")
    print(f"    Batch accuracy:        {exp1['batch_accuracy']:.1f}%")
    print(f"    Per-run accuracy:      {exp1['per_run_accuracy']:.1f}%")
    print(f"    Pairs > 2.0:           {exp1['pairs_above_2']}/105")
    print(f"    Pairs > 3.0:           {exp1['pairs_above_3']}/105")

    print(f"\n  Experiment 2 (Hard Pairs):")
    for label, res in exp2.items():
        print(f"    {label}: ratio={res['ratio']:.2f}, batch={res['batch_accuracy']:.0f}%, per-run={res['per_run_accuracy']:.0f}%")

    print(f"\n  Experiment 3 (Min-Run Sweep):")
    for pair_label, sweep_data in exp3.items():
        for n_fp, mean_acc, std_acc in sweep_data:
            if not np.isnan(mean_acc):
                print(f"    {pair_label} N={n_fp}: {mean_acc:.1f}% +/- {std_acc:.1f}%")

    print(f"\n  Experiment 4 (Cross-Session Stability):")
    print(f"    Same-session:   {exp4['same_session_accuracy']:.1f}%")
    print(f"    Cross-session:  {exp4['cross_session_accuracy']:.1f}%")
    print(f"    Delta:          {exp4['delta']:+.1f}%")

    print(f"\n  Experiment 5 (FAR/FRR/EER):")
    print(f"    EER:            {exp5['eer']*100:.2f}%")
    print(f"    EER threshold:  {exp5['eer_threshold']:.4f}")

    # Overall verdicts
    print("\n  " + "-" * 40)

    if exp1["batch_accuracy"] >= 80 and exp1["per_run_accuracy"] >= 60:
        print("  EXP 1 (Scale):          PASS")
    elif exp1["batch_accuracy"] >= 60:
        print("  EXP 1 (Scale):          MARGINAL")
    else:
        print("  EXP 1 (Scale):          FAIL")

    any_hard_pass = any(r["per_run_accuracy"] >= 70 for r in exp2.values())
    if any_hard_pass:
        print("  EXP 2 (Hard Pairs):     PASS")
    else:
        print("  EXP 2 (Hard Pairs):     MARGINAL")

    if abs(exp4["delta"]) < 10:
        print("  EXP 4 (Cross-Session):  PASS")
    elif abs(exp4["delta"]) < 20:
        print("  EXP 4 (Cross-Session):  MARGINAL")
    else:
        print("  EXP 4 (Cross-Session):  FAIL")

    if exp5["eer"] < 0.10:
        print(f"  EXP 5 (EER):            PASS (EER={exp5['eer']*100:.2f}%)")
    elif exp5["eer"] < 0.20:
        print(f"  EXP 5 (EER):            MARGINAL (EER={exp5['eer']*100:.2f}%)")
    else:
        print(f"  EXP 5 (EER):            FAIL (EER={exp5['eer']*100:.2f}%)")

    print("  " + "-" * 40)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_validation(use_maas)
