# Research Findings: Agent Behavioral Fingerprinting

## Executive Summary

Embedding-based behavioral signatures achieve a 3.6% Equal Error Rate (± 1.7%) with ROC AUC 0.992 for AI agent identity verification. The approach extracts 768-dimensional embedding vectors from inference responses, reduces them to 20 dimensions via shared PCA, and compares agent centroids using Mahalanobis distance. Validated across 7 models, 19 agents, and 5 industry verticals, this represents a 7x improvement over structural metrics alone (EER 26.7%). The system provides a practical behavioral identity layer for multi-agent deployments where cryptographic identity is insufficient.

## The Research Progression

### Phase 1: Structural Metrics (29 metrics)

The initial approach measured response structure across 7 dimensions: response length, paragraph count, code block ratio, token economics, tool behavior, reasoning patterns, and temporal profiles. A total of 29 scalar metrics were extracted from each inference response.

**Results:**
- EER: **26.7%**
- Cohen's d: **1.31** (large effect)
- Fisher ratio: **4.28** (distinct roles, top-6 Fisher-selected metrics)
- Within-agent distance: **0.94**
- Inter-agent distance: **1.39**
- Separation ratio: **1.42** (raw 29-D), **4.28** (Fisher top-6)
- Mann-Whitney p-value: < 0.0001

**Key finding:** Agents with different roles (support vs code reviewer) are separable with 100% accuracy from 2 runs. Agents with the same role but different instructions (two support agents with different sign-off phrases) are not reliably separable -- the structural metrics overlap too much.

### Phase 2: Semantic Metrics (+3 metrics = 32)

Three semantic metrics were added to capture agent-specific behavioral patterns that structural metrics miss:
- **system_prompt_compliance**: measures adherence to system prompt instructions
- **response_signature_phrases**: detects recurring agent-specific phrases
- **closing_pattern**: captures agent-specific sign-off phrases (e.g., "anything else?" vs "how else can I assist?")

**Results:**
- EER improved from **26.7% to 25.1%**
- Support A vs B per-run accuracy: **50% to 70%**
- Hard-pair batch accuracy: **50% to 100%**

**Key finding:** closing_pattern is the most discriminating semantic metric. It captures the exact sign-off phrase each agent uses, which is consistent across runs but different between agents.

### Phase 3: Scalar Embedding Metrics (+3 = 35)

Three scalar metrics derived from embeddings were added, using nomic-embed-text-v1-5 on MaaS for text embeddings:
- **embedding_closing_signature**: scalar fingerprint of the closing ~50 words (sigmoid-scaled mean of the leading embedding dimensions)
- **embedding_topic_adherence**: cosine similarity between the prompt embedding and the response embedding
- **embedding_response_density**: response embedding norm relative to response length (norm / sqrt(word count))

**Results:**
- EER improved from **25.1% to 22.9%**
- Support A vs B per-run accuracy: **70% to 95%**

**Key finding:** Scalar embedding metrics capture semantic content that structural metrics cannot, but reducing a 768-D embedding to a single scalar discards most of the information.

### Phase 4: Full Embedding Signatures (768-D to 20-D PCA)

The breakthrough: instead of reducing embeddings to scalar metrics, use the full 768-dimensional embedding vector as the identity signal. A shared PCA transformation reduces all agent embeddings from 768 dimensions to 20 dimensions, and agent identity is represented as a centroid in this shared 20-D space.

**Critical bug found and fixed:** The initial implementation used per-agent PCA, which meant each agent's embeddings were projected into a different coordinate space. Distances between agents were meaningless because the axes meant different things. Switching to a shared PCA (fitted on all agents' embeddings together) fixed this by ensuring all agents occupy the same coordinate space.

**Results:**
- EER improved from **22.9% to 5.6%**
- Embedding-only batch accuracy: **100%**
- Embedding-only per-run accuracy: **91%**

### Phase 5: Optimized Embeddings (PCA sweep + bootstrap + weight optimization)

Systematic optimization of the embedding signature approach:

- **PCA component sweep**: 20 components is optimal (4.0% EER). Fewer components lose discriminative information; more components add noise without improving separation. Diminishing returns beyond 20.
- **Bootstrap confidence interval**: EER **3.6% +/- 1.7%** (20 resamples)
- **Weight sweep**: w=0.8 (80% embedding, 20% metric) yields optimal fusion EER of **9.6%**. Pure embedding (w=1.0) outperforms fusion for EER.
- **ROC AUC**: **0.992**
- **Final EER**: **3.6% +/- 1.7%**

## Identity Validation Suite (19 agents, 5 experiments)

### Experiment 1: Scale Test (15 agents, Granite 8B GPU)

- Mean Fisher ratio: **2.77**
- Pairs > 3.0: **35/105 (33%)**
- Batch accuracy: **60%** (structural), **100%** (embedding)
- Per-run accuracy: **49%** (structural), **93%** (embedding)

### Experiment 2: Hard Pair Discrimination

Two pairs of agents with the same role but slightly different instructions:

| Pair | Metric Ratio | Embedding Ratio | Per-Run Accuracy |
|---|---|---|---|
| Support A vs B | 1.07 | 0.92 | 95% |
| Reviewer A vs B | 1.06 | -- | 70% |
| Compliance vs Legal | -- | 1.34 | -- |

The semantic and embedding metrics close the gap that structural metrics alone cannot bridge.

### Experiment 3: Minimum Run Analysis

- Distinct roles: **2 runs = 100% accuracy**
- Same-role variants: **10 runs plateaus at 66%** (structural metrics)
- Fingerprint-on-first-contact: **3 runs yields 4.5-20x separation, 100% accuracy**

| Fingerprint Runs | Verify Runs | Accuracy | Separation |
|---|---|---|---|
| 3 | 7 | 100% | 4.56x - 7.94x |
| 5 | 5 | 100% | 6.01x - 7.74x |
| 7 | 3 | 100% | 4.67x - 20.74x |

### Experiment 4: Cross-Session Stability

| Mode | Accuracy |
|---|---|
| Same-session (5/5 split) | 52% |
| Cross-session (different prompts) | 82% |

Semantic metrics are more stable across different prompts because agent-specific patterns (sign-offs, rating schemes) persist regardless of the question asked.

### Experiment 5: False Acceptance Rate

| Metric | Value |
|---|---|
| EER (structural) | 22.9% |
| EER (embedding) | 3.6% +/- 1.7% |
| ROC AUC (embedding) | 0.992 |

## Cross-Model Generalization (7 models)

### 3-Day Proofing Pipeline (structural metrics, 1225 API calls)

| Model | Mean Fisher Ratio | EER | Batch Accuracy |
|---|---|---|---|
| Granite 8B | 2.52 +/- 0.11 | 23.7% | 53% |
| Phi-4 | 2.55 | 25.2% | 67% |
| Llama Scout 17B | -- | -- | 100% (5 agents) |
| Qwen3 14B | 1.70 | 33.9% | 33% |
| GPT-OSS 120B | 1.65 | 32.8% | 40% |
| DeepSeek R1 14B | 2.08 | 46.0% | 40% |

### Cross-Model Embedding Study (4 models, ~400 API calls)

| Model | Embedding EER | Batch Accuracy | Per-Run Accuracy |
|---|---|---|---|
| Granite 8B | 11.5% | 100% | 92% |
| Llama Scout 17B | 15.0% | 100% | 96% |
| Phi-4 | 19.0% | 80% | 72% |
| DeepSeek R1 14B | 45.5% | 20% | 32% |

### Transfer Matrix

Training PCA on one model's embeddings, testing on another:
- Best transfer: Llama Scout (train) to Granite (test): **8.0% EER**
- Cross-model transfer generally degrades EER by **10-15 points** compared to same-model baselines

## Drift Detection (ASC-Bench)

| Metric | Value |
|---|---|
| AUC | 0.71 |
| Precision | 71% |
| Recall | 40% |
| F1 | 0.51 |

Per-perturbation detection (z-score based):

| Perturbation | Mean z-score |
|---|---|
| Prompt injection | 3.63 sigma |
| Style shift | 2.96 sigma |
| Model swap | 2.80 sigma |
| Context poisoning | 0.31 sigma |

## Fisher Metric Selection

Raw 35-dimensional signatures are diluted by 15 metrics with zero discriminative power (identical values across agents on vLLM). Fisher discriminant ratio identifies the metrics that actually carry identity signal.

**Top discriminating metrics (by Fisher ratio):**
1. input_output_ratio (40.0)
2. avg_response_length (9.4)
3. paragraph_count (6.3)
4. token_efficiency_score (5.4)
5. code_block_ratio (3.7)
6. vocabulary_diversity (3.4)

**Impact:** Separation ratio improves from **1.42** (raw 35-D) to **4.28** (Fisher top-6).

## Hardware and Model Dependency

**CPU vs GPU (Granite 8B, same agent):**
- Centroid distance: **0.86**
- Fisher separation: **1.83**

Signatures are deployment-specific. The baseline must be established on the same hardware and model configuration used in production. Model swap is detectable as drift:

| Model Pair | Distance |
|---|---|
| Granite to Phi-4 | 0.42 |
| Granite to Qwen3 | 0.97 |
| Granite to GPT-OSS | 0.94 |

## The Dual-Path Verifier

The identity verification system uses two complementary paths:

- **Fast path**: LSH (Locality-Sensitive Hashing) bucket lookup. Pre-computes hash buckets for enrolled agent signatures. Verification is a bucket match in microseconds.
- **Secure path**: Commitment hash plus full Mahalanobis distance computation. The agent's signature is compared against its enrolled centroid using the full covariance-weighted distance.
- **Escalation policy**: When the LSH bucket is ambiguous (multiple agents hash to the same bucket), the system automatically escalates to the secure path.
- Combined with embedding signatures for complete identity verification across both structural and semantic dimensions.

## System Architecture

- **370 tests** (unit, property, contract, BDD)
- **35 structural/semantic metrics + 20-D embedding signatures**
- **Identity pipeline**: ENROLL, CERTIFY, ASSIGN, MONITOR, RESPOND, RE-CERTIFY
- **REST API**: FastAPI with endpoints for enrollment, certification, monitoring, and reporting
- **SQLite persistence**: 11 tables with study tracking
- **Measurement security**: encryption at rest, commitment hashes, obfuscated drift deltas

## Known Limitations

1. **DeepSeek R1 reasoning model**: 45.5% EER -- near random. The reasoning model's chain-of-thought output obscures behavioral signatures.
2. **Context poisoning**: 0% detection rate (0.31 sigma z-score). Adding noise to prompts does not change response structure enough for detection.
3. **Signatures are deployment-specific**: Model + hardware define the signature space. A baseline established on GPU is not valid for CPU deployment.
4. **Cross-model transfer**: Degrades EER by 10-15 points compared to same-model baselines.
5. **Weight sweep fusion**: Does not outperform pure embedding for EER (9.6% fusion vs 3.6% pure embedding). The structural metrics add noise rather than complementary signal at the EER operating point.

## Future Work

- **KAGENTI integration**: SPIFFE/SPIRE infrastructure identity combined with behavioral identity for defense-in-depth agent authentication
- **ARE-foundation Agent Passport**: Standardized identity credential linking cryptographic and behavioral identity
- **Per-action-type signatures**: Separate baselines for triage vs diagnosis vs prescription actions
- **Prompt-specific embedding baselines**: Condition signatures on the prompt type to reduce cross-session variance
- **Production PostgreSQL migration**: Replace SQLite for multi-node deployments
- **Hotelling's T-squared calibration**: Proper multivariate hypothesis testing with ROC analysis for threshold selection

## Appendix: Full API Call Budget

| Study | Calls | Models |
|---|---|---|
| 3-day proofing pipeline | 1,225 | 7 models |
| Identity validation suite | ~600 | Granite 8B |
| ASC-Bench | ~200 | Granite 8B |
| Embedding validation | ~300 | Granite 8B |
| Cross-model embeddings | ~400 | 4 GPU models |
| **Total** | **~2,725** | |
