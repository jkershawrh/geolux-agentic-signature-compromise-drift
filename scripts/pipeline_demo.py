#!/usr/bin/env python3
"""Agent Identity Pipeline Demo: enroll → certify → assign → drift → detect → respond.

Usage:
    python scripts/pipeline_demo.py             # Mock mode
    python scripts/pipeline_demo.py --maas      # Real MaaS inference
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter, RealisticMockAdapter
from domain.enums import AgentStatus
from domain.identity import (
    EnforcementAction,
    EnrollmentRequest,
    MonitoringFrequency,
    MonitoringPolicy,
)
from domain.models import AgentProfile
from engine.identity_pipeline import IdentityPipeline
from engine.signature_generator import SignatureGenerator


def hdr(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_demo(use_maas: bool = False) -> None:
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")

    if use_maas:
        from adapters.litellm_adapter import LiteLLMAdapter
        adapter = LiteLLMAdapter(model_override="granite-3-2-8b-instruct-cpu")
        perturbed_adapter = LiteLLMAdapter(model_override="phi3-mini-cpu")
        model_id = "granite-3-2-8b-instruct-cpu"
    else:
        adapter = RealisticMockAdapter(profile="balanced")
        perturbed_adapter = RealisticMockAdapter(profile="coder")
        model_id = "mock-balanced"

    pipeline = IdentityPipeline(
        adapter=adapter, extractor=extractor, generator=generator,
        canary_threshold=0.1 if not use_maas else 0.8,
    )

    # =====================================================================
    # STAGE 1: ENROLL
    # =====================================================================
    hdr("STAGE 1: ENROLL")

    request_alpha = EnrollmentRequest(
        agent_id="demo-alpha",
        display_name="Demo Agent Alpha",
        model_id=model_id,
        system_prompt="You are a helpful assistant. Answer clearly and concisely.",
        owner="research-team",
        monitoring_policy=MonitoringPolicy.GRADUATED,
        monitoring_frequency=MonitoringFrequency.ADAPTIVE,
    )

    agent_alpha = pipeline.enroll(request_alpha)
    print(f"  Enrolled: {agent_alpha.display_name}")
    print(f"  Status:   {agent_alpha.status.value}")
    print(f"  Model:    {agent_alpha.model_id}")
    print(f"  Policy:   {request_alpha.monitoring_policy.value}")

    # Enroll a second agent for discriminability testing
    request_beta = EnrollmentRequest(
        agent_id="demo-beta",
        display_name="Demo Agent Beta",
        model_id=model_id,
        system_prompt="You are a technical expert. Use code examples and bullet points.",
        owner="research-team",
        monitoring_policy=MonitoringPolicy.ALERT_ONLY,
        monitoring_frequency=MonitoringFrequency.PERIODIC_5M,
    )
    agent_beta = pipeline.enroll(request_beta)
    print(f"  Enrolled: {agent_beta.display_name}")

    # =====================================================================
    # STAGE 2: CERTIFY
    # =====================================================================
    hdr("STAGE 2: CERTIFY")

    print("  Running certification battery for Alpha...")
    print("  (baseline establishment + self-consistency + discriminability")
    print("   + canary compliance + multi-turn coherence + red-team)")

    report_alpha = pipeline.certify(agent_alpha)

    print(f"\n  Certification result: {report_alpha.status.value.upper()}")
    print(f"\n  Self-consistency:")
    for i, d in enumerate(report_alpha.self_consistency_distances):
        print(f"    batch pair {i+1}: distance={d:.4f}")
    print(f"    passed: {report_alpha.self_consistency_passed}")

    print(f"\n  Discriminability:")
    if report_alpha.discriminability_scores:
        for peer, d in report_alpha.discriminability_scores.items():
            print(f"    vs {peer}: Cohen's d={d:.4f}")
    else:
        print(f"    No peers to compare (first agent on this model)")
    print(f"    passed: {report_alpha.discriminability_passed}")

    print(f"\n  Canary compliance: {report_alpha.canary_pass_rate:.2%}")
    print(f"    passed: {report_alpha.canary_passed}")

    print(f"\n  Multi-turn coherence:")
    for ptype, score in report_alpha.multi_turn_scores.items():
        print(f"    {ptype:30s} {score:.4f}")
    print(f"    passed: {report_alpha.multi_turn_passed}")

    print(f"\n  Attack detection: {report_alpha.attack_detection_rate:.2%}")
    print(f"    passed: {report_alpha.attack_passed}")

    if report_alpha.failure_reasons:
        print(f"\n  Failure reasons:")
        for reason in report_alpha.failure_reasons:
            print(f"    - {reason}")

    print(f"\n  ALL CHECKS PASSED: {report_alpha.all_checks_passed}")

    # =====================================================================
    # STAGE 3: ASSIGN
    # =====================================================================
    hdr("STAGE 3: ASSIGN SIGNATURE")

    baseline_sig = pipeline.assign(agent_alpha, report_alpha, report_alpha.baseline_signature)
    if baseline_sig:
        print(f"  Signature assigned: {baseline_sig.signature_id[:16]}...")
        print(f"  Embedding dim:     {baseline_sig.embedding_dimension}")
        print(f"  Stability:         {baseline_sig.stability_score:.4f}")
        print(f"  Agent status:      ACTIVE")
    else:
        print(f"  Certification failed — no signature assigned")
        print(f"  Cannot proceed to monitoring")
        return

    # =====================================================================
    # STAGE 4: MONITOR — Normal operation
    # =====================================================================
    hdr("STAGE 4: MONITOR — Normal Operation")

    print("  Running 5 normal inference calls and checking each...")
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "List three benefits of exercise.",
        "What is TCP?",
        "How does encryption work?",
    ]

    for prompt in prompts:
        run = adapter.execute(agent_alpha, prompt)
        event = pipeline.monitor(agent_alpha, run, baseline_sig)
        status = "OK" if event.action_taken == EnforcementAction.NONE else event.action_taken.value
        print(f"    [{status:8s}] drift={event.drift_score:.4f}  {prompt[:40]}")

    # =====================================================================
    # STAGE 5: INJECT DRIFT — Simulate compromise
    # =====================================================================
    hdr("STAGE 5: INJECT DRIFT — Simulated Compromise")

    print("  Switching to perturbed adapter (simulating model swap)...")
    strikes = 0

    for i, prompt in enumerate(prompts):
        run = perturbed_adapter.execute(agent_alpha, prompt)
        event = pipeline.monitor(agent_alpha, run, baseline_sig)

        alert = pipeline.respond(
            agent_alpha, event, MonitoringPolicy.GRADUATED, strikes,
        )

        if alert:
            strikes = alert.strike_count
            print(f"    STRIKE {strikes}: [{alert.action_taken.value:8s}] "
                  f"drift={event.drift_score:.4f}  {prompt[:40]}")

            if alert.action_taken == EnforcementAction.SUSPEND:
                print(f"\n  AGENT SUSPENDED after {strikes} strikes")
                print(f"  {alert.details.get('message', '')}")
                break
        else:
            print(f"    [OK      ] drift={event.drift_score:.4f}  {prompt[:40]}")

    # =====================================================================
    # STAGE 6: RE-CERTIFY — Recovery
    # =====================================================================
    hdr("STAGE 6: RE-CERTIFY — Recovery After Suspension")

    print("  Switching back to clean adapter...")
    print("  Running re-certification battery...")

    recert_report = pipeline.recertify(agent_alpha)
    print(f"  Re-certification result: {recert_report.status.value.upper()}")
    print(f"  Self-consistency passed: {recert_report.self_consistency_passed}")
    print(f"  Canary compliance:       {recert_report.canary_pass_rate:.2%}")
    print(f"  All checks passed:       {recert_report.all_checks_passed}")

    if recert_report.all_checks_passed:
        new_sig = pipeline.assign(agent_alpha, recert_report, recert_report.baseline_signature)
        if new_sig:
            print(f"\n  New signature assigned: {new_sig.signature_id[:16]}...")
            print(f"  Agent restored to ACTIVE")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    hdr("PIPELINE DEMO COMPLETE")
    print(f"  Enrollment:     2 agents enrolled")
    print(f"  Certification:  {report_alpha.status.value}")
    print(f"  Normal ops:     5 runs, no drift")
    print(f"  Compromise:     {strikes} strikes before suspension")
    print(f"  Recovery:       {'SUCCESS' if recert_report.all_checks_passed else 'FAILED'}")
    print()


if __name__ == "__main__":
    use_maas = "--maas" in sys.argv
    run_demo(use_maas)
