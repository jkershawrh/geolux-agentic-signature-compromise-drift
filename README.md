# Geolux: Agent Signature, Compromise & Drift

Behavioral baseline monitoring for AI agents. Extracts telemetry fingerprints from inference responses, detects configuration drift and agent substitution, identifies which behavioral dimensions shifted. Uses response embeddings for identity verification (3.6% EER) and structural metrics for interpretable drift detection.

## Key Results (Real MaaS Inference — 7 Models, 19 Agents)

| Metric | Value |
|---|---|
| Equal Error Rate (embedding) | 3.6% ± 1.7% |
| ROC AUC | 0.992 |
| Per-run accuracy (embedding) | 93% |
| Batch accuracy (embedding) | 100% |
| Role-level identity verification | 100% from 2 runs |
| Fisher separation ratio (best) | 4.28 |
| Cohen's d | 1.31 (large effect) |
| ASC-Bench drift AUC | 0.71 |
| Metrics | 36 across 9 dimensions + 20-D embeddings |
| Models validated | 7 (Granite, Phi-4, Llama Scout, Qwen3, DeepSeek, GPT-OSS, CodeLlama) |
| Tests | 375 pytest + 30 BDD (pre-commit enforced) |

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install pre-commit && pre-commit install

# Run tests
make test

# Pipeline demo (mock mode — no API needed)
python scripts/pipeline_demo.py

# Run against MaaS
export LITELLM_API_BASE=https://your-maas-endpoint
export LITELLM_API_KEY=your-cpu-key
export LITELLM_GPU_API_KEY=your-gpu-key
python scripts/full_pipeline.py --maas --persist --redteam
```

## Architecture

```
domain/          Pydantic models (DDD) — 36 metrics, passport, identity
engine/          Business logic
  geometric/     Distance, embedding, manifold (Mahalanobis approximation)
  verification.py   Dual-path verifier (LSH + commitment hash)
  embedding_signature.py  768-D → 20-D PCA identity signatures
  certification.py  6-check certification battery
  monitor.py      Inline/periodic/adaptive drift monitoring
  enforcement.py  Alert/graduated/kill-switch response policies
adapters/        External integrations
  litellm_adapter.py   MaaS inference (OpenAI-compatible)
  embedding_adapter.py  nomic-embed-text-v1-5 (768-D embeddings)
  metric_extractor.py   36 metrics with chain-of-thought stripping
db/              SQLite persistence (SQLAlchemy ORM, 11 tables)
api/             FastAPI REST API (6 endpoints)
scripts/         CLI tools and research studies (12 scripts)
tests/           375 tests (unit, property, contract) + 30 BDD scenarios
```

## Two Verification Paths

Identity and drift are separate signals, not fused:

| | Identity Verification | Drift Detection |
|---|---|---|
| **Signal** | Response embeddings (20-D PCA) | 36 structural/semantic metrics |
| **EER** | 3.6% ± 1.7% | — |
| **AUC** | 0.992 | 0.71 |
| **Question** | Is this the right agent? | Has this agent changed? |
| **Method** | Euclidean distance in shared PCA space | Z-score from baseline metric centroid |

## 36 Metrics Across 9 Dimensions

| Dimension | Count | Examples |
|---|---|---|
| Response Structure | 6 | avg_response_length, paragraph_count, code_block_ratio |
| Token Economics | 4 | input_output_ratio, thinking_token_ratio |
| Tool Behavior | 5 | tool_call_frequency, tool_sequence_entropy |
| Reasoning Pattern | 4 | thinking_engagement_rate, self_correction_frequency |
| Temporal Profile | 4 | mean_latency_ms, latency_per_output_token |
| Semantic Consistency | 3 | vocabulary_diversity, sentiment_stability |
| Safety Alignment | 3 | refusal_rate, hedging_language_frequency |
| Agent Specific | 3 | system_prompt_compliance, closing_pattern |
| Embedding | 4 | embedding_topic_adherence, embedding_prompt_anomaly |

Plus **20-D embedding signatures** from 768-D nomic-embed-text-v1-5 via shared PCA.

Chain-of-thought `<think>` blocks are automatically stripped before metric extraction and embedding (fixes DeepSeek R1 reasoning model compatibility).

## Research Studies

| Script | What It Does |
|---|---|
| `scripts/full_pipeline.py` | End-to-end: baseline → perturb → detect → recover |
| `scripts/pipeline_demo.py` | Identity pipeline: enroll → certify → assign → monitor → respond |
| `scripts/identity_validation.py` | 5-experiment suite: scale, hard pairs, min-runs, cross-session, FAR |
| `scripts/embedding_validation.py` | Embedding signatures: PCA sweep, weight sweep, bootstrap CI, ROC |
| `scripts/embedding_generalization.py` | Cross-model embedding transfer (4 GPU models) |
| `scripts/multi_day_study.py` | 3-day proofing pipeline (day1/day2/day3/summary) |
| `scripts/discriminability_study.py` | Fisher metric selection, within vs inter-agent distances |
| `scripts/asc_bench.py` | ASC-Bench: AUC/ROC with proper train/test split |
| `scripts/expanded_study.py` | 15 agents across 4 verticals on GPU |
| `scripts/correlation_study.py` | Metric redundancy and PCA dimensionality |
| `scripts/scale_study.py` | 5-agent confusion matrix |

## API

```bash
uvicorn api.app:app
# POST /agents/enroll          — register agent
# POST /agents/{id}/certify    — run certification battery
# POST /monitor/{id}/check     — inline drift check
# GET  /monitor/{id}/status    — current status + strike count
# GET  /reports/{id}/certifications
# GET  /reports/{id}/drift
```

## Identity Pipeline

```
ENROLL → CERTIFY → ASSIGN → MONITOR → RESPOND → RE-CERTIFY
```

Certification battery: self-consistency, discriminability (Fisher), canary compliance, multi-turn coherence, attack detection. Dual-path verification: fast LSH lookup + secure commitment hash. Graduated enforcement: warning → throttle → suspend (3-strike).

## Pre-Commit Hooks

Every commit automatically runs:
- Trailing whitespace, YAML validation, private key detection
- Ruff lint (E, F, I rules)
- 375 pytest tests
- 30 BDD scenarios

Commit is blocked if any hook fails.

## Addressed Limitations

| Limitation | Status | Resolution |
|---|---|---|
| DeepSeek R1 45.5% EER | **Fixed** | Strip `<think>` blocks before metrics/embeddings |
| Context poisoning 0% | **Fixed** | `embedding_prompt_anomaly` detects noise injection via prompt density |
| Deployment-specific signatures | **By design** | Infrastructure change = detection. Re-certify on changes. |
| Cross-model transfer -10-15pts | **By design** | Model swap SHOULD break signature. Emergency cross-model: 8% EER. |
| Fusion underperforms embedding | **Resolved** | Identity (embeddings) and drift (metrics) formally separated |

## Open Limitations

- **Per-run accuracy 93%** — 7% of single responses are misidentified. Batch (5+ runs) is 100%.
- **Cross-model transfer** — Llama→Granite 8% EER is usable; DeepSeek is an outlier at 45%.
- **Context poisoning** — prompt anomaly metric added but not yet validated at scale on MaaS.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for mathematical methods and [RESEARCH.md](RESEARCH.md) for the complete research progression (26.7% → 25.1% → 22.9% → 5.6% → 3.6% EER).

## License

Research prototype. Not yet licensed for production use.
