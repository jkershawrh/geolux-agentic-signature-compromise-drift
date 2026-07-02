# Behavioral Telemetry for Agent Degradation Detection

## The Problem

AI agents can degrade silently. The model doesn't crash — it keeps responding. But responses are subtly wrong: hallucinated facts, ignored instructions, shifted tone, degraded reasoning. Infrastructure monitoring (uptime, latency, error rates) sees nothing. SPIFFE confirms the container is authentic. The agent is broken and nobody knows.

## What We Built

A behavioral fingerprinting layer that extracts 36 metrics from every agent response and detects when behavior shifts from an established baseline. It answers two questions:

- **Identity**: Is this still the same agent? (embedding-based, 3.6% EER)
- **Drift**: Has this agent's behavior changed? (metric-based, per-dimension decomposition)

## How It Works

```
Agent response → Extract 36 metrics → Compare to baseline → Alert if shifted
                 + Embed response (768-D) → PCA → Compare centroid → Verify identity
```

**36 metrics across 9 behavioral dimensions**: response structure, token economics, tool behavior, reasoning patterns, temporal profile, semantic consistency, safety alignment, agent-specific compliance, embedding signatures.

**No model access required.** Works with any OpenAI-compatible API. Observe the response, extract telemetry, compare to baseline. The agent doesn't know it's being monitored.

## Key Results (Real Inference — 7 Models, 19 Agents)

| What We Measured | Result |
|---|---|
| Can we tell agents apart? | **3.6% EER** (96.4% correct), **AUC 0.992** |
| How fast to fingerprint? | **3 runs** to establish identity |
| Per-response accuracy? | **93%** from a single response |
| Does it generalize? | Validated on **Granite 8B, Phi-4, Llama Scout 17B, Qwen3, DeepSeek, GPT-OSS** |
| Confidence intervals? | EER **3.6% ± 1.7%** (20 bootstrap resamples) |

## What It Catches

| Degradation Type | Detection | Signal |
|---|---|---|
| Model swap (Granite → Phi-3) | Detected | Temporal profile + token economics shift |
| Prompt injection | Detected | Response structure + semantic consistency shift |
| Style degradation | Detected | Vocabulary diversity + response length shift |
| Instruction drift | Detected | Agent-specific compliance drops |
| Context poisoning | Partial | Prompt embedding anomaly (new metric) |

## Chaos Engineering Connection

A chaos engineering framework **injects** failures. This system **measures** the behavioral impact. Together:

```
Chaos engine injects perturbation
  → Agent produces degraded response
    → Behavioral telemetry detects the shift
      → Per-dimension decomposition shows WHAT changed
        → Severity scored automatically
```

The chaos framework needs a scoring function that goes beyond "did it crash?" This is that function. It quantifies *how much* behavior changed and *which dimensions* shifted — even when the agent keeps running.

## Technical Details

- **36 metrics** extracted per response (no inference cost — text analysis + optional embeddings)
- **Fisher discriminant selection** identifies the 6 most discriminating metrics per agent pair
- **Shared PCA** on 768-D response embeddings for identity verification
- **Dual-path verification**: fast LSH lookup (microseconds) + secure commitment hash
- **382 tests**, validated across 7 model families, 19 agents, 5 industry verticals
- Self-contained Python package, SQLite persistence, FastAPI REST API

## What's Next

Looking for a deployment to validate against real agent workloads — 3 agents, 1 week, behavioral fingerprinting running alongside existing observability. The research is proven on MaaS inference; production validation is the next step.

---

*Jonathan Kershaw · maintainer@example.com*
