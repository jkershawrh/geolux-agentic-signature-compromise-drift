# Geolux: Agent Signature, Compromise & Drift

Behavioral baseline monitoring for AI agents. Extracts telemetry fingerprints from inference responses, detects configuration drift and agent substitution, identifies which behavioral dimensions shifted.

## Key Results (Real MaaS Inference — Granite 8B GPU)

| Metric | Value |
|---|---|
| Role-level identity verification | 100% from 2 runs |
| Fisher separation ratio (best) | 4.28 |
| Cohen's d | 1.31 (large effect) |
| Mann-Whitney p-value | < 0.0001 |
| ASC-Bench drift AUC | 0.71 |
| Equal Error Rate | 25% |
| Metrics | 32 across 8 dimensions |
| Tests | 354 pytest + 30 BDD |

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
db/              SQLite persistence (SQLAlchemy ORM, 10 tables)
api/             FastAPI REST API
scripts/         CLI tools and research studies
tests/           354 tests (unit, property, contract, BDD)
rubrics/         Red/green evaluation matrices (13 stages)
visualizations/  Generated plots (gitignored)
```

## 32 Metrics Across 8 Dimensions

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

## Research Studies

| Script | What It Does |
|---|---|
| `scripts/full_pipeline.py` | End-to-end: baseline → perturb → detect → recover |
| `scripts/pipeline_demo.py` | Identity pipeline: enroll → certify → assign → monitor → respond |
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
ENROLL → CERTIFY → ASSIGN → MONITOR → RESPOND → RE-CERTIFY
```

Certification battery: self-consistency, discriminability (Fisher), canary compliance, multi-turn coherence, attack detection. Dual-path verification: fast LSH lookup + secure commitment hash.

## Known Limitations

- **EER 25%**: Per-instance identity within the same role is hard. Two support agents with different sign-off phrases: 70% per-run, 100% batch.
- **Per-run accuracy 47%**: Single-run identification across 15 agents is unreliable. Batch (5+ runs) is required.
- **Signatures are hardware-dependent**: CPU vs GPU produces different signatures. Baseline must be established on production hardware.
- **Signatures are model-dependent**: Same agent on different models produces different signatures.
- **Context poisoning undetectable**: Adding noise to prompts doesn't change response structure enough for detection.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for honest documentation of mathematical approximations. The system uses Mahalanobis distance (not true Riemannian geodesic), arithmetic mean with gradient refinement (not Frechet mean with exp/log maps), and eigenvalue variance (not sectional curvature).

## License

Research prototype. Not yet licensed for production use.
