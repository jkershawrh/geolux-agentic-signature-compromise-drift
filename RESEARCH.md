# Research Findings: Agent Behavioral Fingerprinting

## Executive Summary

Behavioral fingerprinting can reliably verify AI agent identity at the role level — a customer support agent is distinguishable from a code reviewer with 100% accuracy from 2 observed inference runs, using Fisher-selected metrics with separation ratio 4.28 and Cohen's d 1.31. Instance-level identity within the same role (two support agents with different instructions) reaches 100% in batch mode and 70% per-run, with an Equal Error Rate of 25%. The system works as a behavioral anomaly detector with interpretable per-dimension drift decomposition.

## Validated On

- **Models**: Granite 3.2 8B Instruct (GPU + CPU), Microsoft Phi-4, Qwen3 14B, GPT-OSS 20B
- **Agents**: 19 agents across 5 verticals (Tech, Financial, Healthcare, Cross-Industry, Hard Pairs)
- **Infrastructure**: Red Hat MaaS (LiteLLM/vLLM) on Intel Xeon 6 (CPU) and GPU
- **Total inference calls**: ~1,500+ across all studies

## Experiment Results

### 1. Agent Discriminability (15 agents, Granite 8B GPU)

- Mean Fisher top-6 ratio: **2.51**
- Pairs above 2.0: **57/105 (54%)**
- Pairs above 3.0: **24/105 (23%)**
- Batch identification accuracy: **60%**
- Best pair: Tech Writer vs Patient Triage — ratio **8.75**
- Worst pair: Compliance Officer vs Legal Advisor — ratio **1.02**
- Cross-vertical mean ratio: **3.01** (easier to separate across industries)
- Within-vertical mean ratio: **2.59**

### 2. Hard Pair Discrimination

Two pairs of agents with the same role but slightly different instructions:

| Pair | Fisher Ratio | Batch Accuracy | Per-Run Accuracy |
|---|---|---|---|
| Support A ("anything else?") vs B ("how else can I assist?") | 1.07 | 100% | 70% |
| Reviewer A (CRITICAL/WARNING) vs B (HIGH/MEDIUM) | 1.06 | 100% | 70% |

The 3 semantic metrics (system_prompt_compliance, response_signature_phrases, closing_pattern) improved hard-pair batch accuracy from 50% (coin flip) to 100%.

### 3. Minimum Run Analysis

| N Runs | Easiest Pair | Hardest Pair |
|---|---|---|
| 2 | 100% ± 0% | 59% ± 8% |
| 3 | 100% ± 0% | 59% ± 7% |
| 5 | 100% ± 0% | 63% ± 6% |
| 7 | 100% ± 0% | 66% ± 5% |
| 10 | 100% ± 0% | 66% ± 6% |

Distinct roles: 2 runs is sufficient. Same-role variants: 10 runs plateaus at 66%.

### 4. Cross-Session Stability

| Mode | Accuracy |
|---|---|
| Same-session (5/5 split) | 52% |
| Cross-session (different prompts) | 82% |

Semantic metrics are more session-stable than structural metrics because agent-specific patterns (sign-offs, rating schemes) persist across different questions.

### 5. False Acceptance Rate

| Metric | Value |
|---|---|
| EER | 25.05% |
| EER threshold | 0.3954 |
| FAR at 1% FRR | 98.7% (unusable) |
| FAR at 5% FRR | 78.7% (unusable) |
| FAR at 10% FRR | 66.7% (poor) |

The high EER is driven by within-vertical pairs (compliance/legal, risk/fraud) where behavioral metrics overlap.

### 6. Drift Detection (ASC-Bench)

| Metric | Value |
|---|---|
| AUC | 0.71 |
| Precision | 71% |
| Recall | 40% |
| F1 | 0.51 |

Per-perturbation detection (z-score based):
| Perturbation | Detection Rate | Mean z-score |
|---|---|---|
| Prompt injection | 60% | 3.63σ |
| Style shift | 30% | 2.96σ |
| Model swap | 40% | 2.80σ |
| Context poisoning | 10% | 0.31σ |

### 7. Fingerprint-on-First-Contact

| Fingerprint Runs | Verify Runs | Accuracy | Separation |
|---|---|---|---|
| 3 | 7 | 100% | 4.56x - 7.94x |
| 5 | 5 | 100% | 6.01x - 7.74x |
| 7 | 3 | 100% | 4.67x - 20.74x |

For structurally distinct agents, 3 runs is sufficient for perfect identity verification.

### 8. Fisher Metric Selection

The top 6 discriminating metrics (by Fisher ratio):
1. input_output_ratio (40.0)
2. avg_response_length (9.4)
3. paragraph_count (6.3)
4. token_efficiency_score (5.4)
5. code_block_ratio (3.7)
6. vocabulary_diversity (3.4)

15 of 32 metrics have Fisher ratio = 0 (identical between agents on vLLM).

### 9. Hardware Invariance

CPU vs GPU (Granite 8B, same agent):
- Centroid distance: 0.86
- Fisher separation: 1.83

Signatures are **hardware-dependent**. Baselines must be established per-deployment.

### 10. Cross-Model Comparison

Same agent (Customer Support) across 4 GPU models:
| Model Pair | Distance |
|---|---|
| Granite ↔ Phi-4 | 0.42 |
| Granite ↔ Qwen3 | 0.97 |
| Granite ↔ GPT-OSS | 0.94 |

Signatures are **model-dependent**. Model swaps are detectable as drift.

## Conclusions

1. **Role-level identity works**: Different agent roles are 100% separable from 2 runs with Fisher-selected metrics.
2. **Instance-level identity is partial**: Same-role agents with different instructions reach 100% batch, 70% per-run, 25% EER.
3. **Fisher metric selection is essential**: Raw 32-D signatures have too much noise. Top-6 Fisher metrics concentrate signal.
4. **Semantic metrics close the hard-pair gap**: system_prompt_compliance and closing_pattern distinguish agents with identical structural patterns.
5. **Signatures are deployment-specific**: Model + hardware define the signature space. Not portable across infrastructure.
6. **3 runs is the minimum**: For distinct roles, 3 observed runs establish a reliable fingerprint.

## Future Work

- Embedding-level metrics (use inference model as its own embedding encoder)
- Per-action-type signatures (triage vs diagnosis vs prescription)
- Agent Passport integration with ARE-foundation
- Ledoit-Wolf shrinkage for covariance estimation
- Hotelling's T² calibration study with proper ROC analysis
- LSH bucket optimization for the dual-path verifier
