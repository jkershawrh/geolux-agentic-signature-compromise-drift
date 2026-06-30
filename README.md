# Geolux: Agent Signature, Compromise & Drift

Behavioral baseline monitoring for AI agents. Extracts telemetry fingerprints from inference responses, detects configuration drift and agent substitution, identifies which behavioral dimensions shifted.

## Key Results (Real MaaS Inference — 7 Models, 19 Agents)

| Metric | Value |
|---|---|
| Equal Error Rate (embedding) | 3.6% +/- 1.7% |
| ROC AUC | 0.992 |
| Per-run accuracy (embedding) | 93% |
| Batch accuracy (embedding) | 100% |
| Role-level identity verification | 100% from 2 runs |
| Fisher separation ratio (best) | 4.28 |
| Cohen's d | 1.31 (large effect) |
| ASC-Bench drift AUC | 0.71 |
| Metrics | 35 across 8 dimensions + 20-D embeddings |
| Models validated | 7 |
| Tests | 370 (pytest + BDD) |

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
make test

# Pipeline demo (mock mode — no API needed)
python scripts/pipeline_demo.py

# Run against MaaS
export LITELLM_API_BASE=https://your-maas-endpoint
export LITELLM_API_KEY=your-key
python scripts/full_pipeline.py --maas --persist --redteam
```

## Architecture

```
domain/          Pydantic models (DDD)
engine/          Business logic (signature, drift, auth, certification)
  geometric/     Distance, embedding, manifold, Riemannian approximations
adapters/        External integrations (LiteLLM, mock, metric extraction)
db/              SQLite persistence (SQLAlchemy ORM, 11 tables)
api/             FastAPI REST API
scripts/         CLI tools and research studies
tests/           370 tests (unit, property, contract, BDD)
rubrics/         Red/green evaluation matrices (13 stages)
visualizations/  Generated plots (gitignored)
```

## 35 Metrics Across 8 Dimensions + 20-D Embedding Signatures

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
| Embedding Derived | 3 | embedding_closing_signature, embedding_topic_adherence, embedding_response_density |
| **Embedding Signature** | **20-D** | **768-D nomic-embed-text-v1-5 reduced via shared PCA** |

## Research Studies

| Script | What It Does |
|---|---|
| `scripts/full_pipeline.py` | End-to-end: baseline, perturb, detect, recover |
| `scripts/pipeline_demo.py` | Identity pipeline: enroll, certify, assign, monitor, respond |
| `scripts/discriminability_study.py` | Within-agent vs inter-agent distance analysis |
| `scripts/correlation_study.py` | Metric redundancy and PCA dimensionality |
| `scripts/identity_validation.py` | 5-experiment validation suite (scale, hard pairs, min-runs, cross-session, FAR) |
| `scripts/expanded_study.py` | 15 agents across 4 verticals on GPU |
| `scripts/asc_bench.py` | ASC-Bench: AUC/ROC with train/test split |
| `scripts/scale_study.py` | 5-agent confusion matrix |

## API

```bash
uvicorn api.app:app
# Endpoints:
# POST /agents/enroll
# POST /agents/{id}/certify
# POST /monitor/{id}/check
# GET  /monitor/{id}/status
# GET  /reports/{id}/certifications
# GET  /reports/{id}/drift
```

## Identity Pipeline

```
ENROLL -> CERTIFY -> ASSIGN -> MONITOR -> RESPOND -> RE-CERTIFY
```

Certification battery: self-consistency, discriminability (Fisher), canary compliance, multi-turn coherence, attack detection. Dual-path verification: fast LSH lookup + secure commitment hash. Embedding signatures provide the primary identity signal (3.6% EER), with structural metrics providing interpretable drift decomposition.

## Known Limitations

- **DeepSeek R1 near-random**: Reasoning models produce 45.5% EER -- chain-of-thought output obscures behavioral signatures.
- **Context poisoning undetectable**: 0% detection rate (0.31 sigma). Adding noise to prompts does not change response structure enough for detection.
- **Signatures are deployment-specific**: Model + hardware define the signature space. A baseline on GPU is invalid for CPU. Baseline must match production.
- **Cross-model transfer degrades**: EER worsens by 10-15 points when PCA is trained on one model and tested on another.
- **Fusion doesn't beat pure embedding**: Weight-optimized fusion (9.6% EER) underperforms pure embedding (3.6% EER) at the equal error rate operating point.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for documentation of mathematical methods, including embedding signature extraction (768-D to 20-D shared PCA), EER computation, bootstrap confidence intervals, Ledoit-Wolf shrinkage, Fisher discriminant ratio, and the Riemannian approximations used in the geometric engine.

## License

Research prototype. Not yet licensed for production use.
