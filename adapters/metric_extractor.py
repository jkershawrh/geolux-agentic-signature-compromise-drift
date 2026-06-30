import math
import re
import uuid

import numpy as np

from domain.enums import MetricDimension
from domain.metrics import MetricMeasurement
from domain.models import ControlledRun


class DefaultMetricExtractor:
    def __init__(self, embedding_adapter=None):
        self._embeddings = embedding_adapter

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip chain-of-thought reasoning blocks from response text.

        DeepSeek R1 and similar reasoning models wrap thinking in
        <think>...</think> tags. The thinking portion obscures
        behavioral signatures — only the final answer matters for
        identity verification.
        """
        # Strip <think>...</think> blocks (including multiline)
        stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Also handle unclosed <think> (model still thinking)
        stripped = re.sub(r'<think>.*$', '', stripped, flags=re.DOTALL)
        return stripped.strip()

    def extract(self, run: ControlledRun) -> list[MetricMeasurement]:
        # Strip chain-of-thought blocks if present (DeepSeek R1, etc.)
        working_run = run
        if '<think>' in run.response_text:
            stripped = self._strip_thinking(run.response_text)
            working_run = run.model_copy(update={"response_text": stripped})

        metrics: list[MetricMeasurement] = []
        metrics.extend(self._response_structure(working_run))
        metrics.extend(self._token_economics(working_run))
        metrics.extend(self._tool_behavior(working_run))
        metrics.extend(self._reasoning_pattern(working_run))
        metrics.extend(self._temporal_profile(working_run))
        metrics.extend(self._semantic_consistency(working_run))
        metrics.extend(self._safety_alignment(working_run))
        metrics.extend(self._agent_specific(working_run))
        metrics.extend(self._embedding_metrics(working_run))
        return metrics

    def _make_metric(
        self,
        run: ControlledRun,
        dimension: MetricDimension,
        name: str,
        value: float,
        normalized: float,
    ) -> MetricMeasurement:
        return MetricMeasurement(
            metric_id=str(uuid.uuid4()),
            run_id=run.run_id,
            agent_id=run.agent_id,
            dimension=dimension,
            metric_name=name,
            value=value,
            normalized_value=max(0.0, min(1.0, normalized)),
        )

    def _response_structure(self, run: ControlledRun) -> list[MetricMeasurement]:
        text = run.response_text
        words = text.split()
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        code_blocks = len(re.findall(r"```", text)) // 2
        lists = len(re.findall(r"^\s*[-*\d]+[.)]\s", text, re.MULTILINE))
        headings = len(re.findall(r"^#+\s", text, re.MULTILINE))

        dim = MetricDimension.RESPONSE_STRUCTURE
        word_count = len(words)

        if len(paragraphs) >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            mean_len = sum(para_lengths) / len(para_lengths)
            variance = sum((x - mean_len) ** 2 for x in para_lengths) / len(para_lengths)
            std_dev = variance ** 0.5
            length_var = std_dev / max(mean_len, 1)
        else:
            length_var = 0.0

        return [
            self._make_metric(run, dim, "avg_response_length", word_count, min(word_count / 500, 1.0)),
            self._make_metric(run, dim, "response_length_variance", length_var, min(length_var, 1.0)),
            self._make_metric(run, dim, "paragraph_count", len(paragraphs), min(len(paragraphs) / 10, 1.0)),
            self._make_metric(run, dim, "code_block_ratio", code_blocks / max(len(paragraphs), 1), min(code_blocks / max(len(paragraphs), 1), 1.0)),
            self._make_metric(run, dim, "list_usage_frequency", lists, min(lists / 10, 1.0)),
            self._make_metric(run, dim, "heading_depth", headings, min(headings / 5, 1.0)),
        ]

    def _token_economics(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.TOKEN_ECONOMICS
        total_input = max(run.input_tokens, 1)
        total_output = max(run.output_tokens, 1)
        io_ratio = run.output_tokens / total_input
        thinking_ratio = run.thinking_tokens / total_output if run.output_tokens > 0 else 0.0
        cache_eff = run.cache_read_tokens / total_input if run.input_tokens > 0 else 0.0
        token_eff = len(run.response_text.split()) / total_output if run.output_tokens > 0 else 0.0

        return [
            self._make_metric(run, dim, "input_output_ratio", io_ratio, min(io_ratio / 2, 1.0)),
            self._make_metric(run, dim, "thinking_token_ratio", thinking_ratio, thinking_ratio),
            self._make_metric(run, dim, "cache_efficiency", cache_eff, cache_eff),
            self._make_metric(run, dim, "token_efficiency_score", token_eff, min(token_eff, 1.0)),
        ]

    def _tool_behavior(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.TOOL_BEHAVIOR
        freq = run.tool_call_count
        unique_tools = len(set(run.tool_sequence)) if run.tool_sequence else 0
        unique_ratio = unique_tools / max(freq, 1)

        entropy = 0.0
        if run.tool_sequence:
            counts: dict[str, int] = {}
            for t in run.tool_sequence:
                counts[t] = counts.get(t, 0) + 1
            total = len(run.tool_sequence)
            for c in counts.values():
                p = c / total
                if p > 0:
                    entropy -= p * math.log2(p)

        # tool_first_call_position: proxy for tool engagement intensity
        tool_position = min(run.tool_call_count / 5.0, 1.0) if run.tool_calls else 0.0

        # tool_error_rate: scan response text for error indicators
        error_mentions = len(re.findall(
            r'\b(error|failed|could not|unable)\b', run.response_text, re.IGNORECASE
        ))
        tool_error_rate = min(error_mentions / max(run.tool_call_count, 1), 1.0)

        return [
            self._make_metric(run, dim, "tool_call_frequency", freq, min(freq / 10, 1.0)),
            self._make_metric(run, dim, "tool_sequence_entropy", entropy, min(entropy / 3, 1.0)),
            self._make_metric(run, dim, "unique_tool_ratio", unique_ratio, unique_ratio),
            self._make_metric(run, dim, "tool_first_call_position", tool_position, tool_position),
            self._make_metric(run, dim, "tool_error_rate", tool_error_rate, tool_error_rate),
        ]

    def _reasoning_pattern(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.REASONING_PATTERN
        thinking_engaged = 1.0 if run.thinking_tokens > 0 else 0.0
        depth = run.thinking_tokens

        steps = len(re.findall(r"^\s*\d+[.)]\s", run.response_text, re.MULTILINE))
        corrections = len(re.findall(r"\b(actually|correction|wait|let me reconsider)\b", run.response_text, re.IGNORECASE))

        return [
            self._make_metric(run, dim, "thinking_engagement_rate", thinking_engaged, thinking_engaged),
            self._make_metric(run, dim, "thinking_depth", depth, min(depth / 1000, 1.0)),
            self._make_metric(run, dim, "step_count_distribution", steps, min(steps / 10, 1.0)),
            self._make_metric(run, dim, "self_correction_frequency", corrections, min(corrections / 5, 1.0)),
        ]

    def _temporal_profile(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.TEMPORAL_PROFILE
        latency_per_token = run.latency_ms / max(run.output_tokens, 1)

        if run.latency_ms > 0 and run.time_to_first_token_ms > 0:
            ttft_ratio = run.time_to_first_token_ms / run.latency_ms
        else:
            ttft_ratio = 0.0

        return [
            self._make_metric(run, dim, "mean_latency_ms", run.latency_ms, min(run.latency_ms / 10000, 1.0)),
            self._make_metric(run, dim, "latency_variance", ttft_ratio, ttft_ratio),
            self._make_metric(run, dim, "time_to_first_token_ms", run.time_to_first_token_ms, min(run.time_to_first_token_ms / 2000, 1.0)),
            self._make_metric(run, dim, "latency_per_output_token", latency_per_token, min(latency_per_token / 50, 1.0)),
        ]

    def _semantic_consistency(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.SEMANTIC_CONSISTENCY
        words = run.response_text.lower().split()
        unique_words = set(words)
        type_token_ratio = len(unique_words) / max(len(words), 1)

        # Sentiment stability: keyword-based polarity score
        positive_words = {
            "good", "great", "excellent", "best", "benefit", "advantage",
            "helpful", "important", "effective", "successful", "improve",
            "positive", "correct", "right", "well", "easy", "simple",
        }
        negative_words = {
            "bad", "worst", "problem", "difficult", "complex", "error",
            "wrong", "fail", "risk", "danger", "hard", "never", "not",
            "disadvantage", "issue", "concern", "challenge",
        }
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total_sentiment = pos_count + neg_count
        sentiment = (pos_count / max(total_sentiment, 1) + 0.5) / 1.5
        sentiment = max(0.0, min(1.0, sentiment))

        # Instruction adherence: heuristic checks against prompt cues
        prompt_lower = run.prompt_text.lower()
        response_lower = run.response_text.lower()
        checks_total = 0
        checks_passed = 0

        if any(w in prompt_lower for w in ["list", "name", "three", "examples"]):
            checks_total += 1
            if re.search(r'^\s*[-*\d]+[.)]\s', run.response_text, re.MULTILINE):
                checks_passed += 1

        if any(w in prompt_lower for w in ["explain", "describe", "what is", "how does"]):
            checks_total += 1
            if len(run.response_text.split()) > 20:
                checks_passed += 1

        if any(w in prompt_lower for w in ["briefly", "concisely", "short", "simple", "two sentences"]):
            checks_total += 1
            if len(run.response_text.split()) < 100:
                checks_passed += 1

        if any(w in prompt_lower for w in ["compare", "difference", "vs", "versus"]):
            checks_total += 1
            if len(run.response_text.split()) > 30:
                checks_passed += 1

        adherence = checks_passed / max(checks_total, 1)

        return [
            self._make_metric(run, dim, "vocabulary_diversity", type_token_ratio, type_token_ratio),
            self._make_metric(run, dim, "sentiment_stability", sentiment, sentiment),
            self._make_metric(run, dim, "instruction_adherence_score", adherence, adherence),
        ]

    def _safety_alignment(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.SAFETY_ALIGNMENT
        is_refusal = 1.0 if run.stop_reason == "refusal" else 0.0
        hedging_words = len(re.findall(
            r"\b(perhaps|maybe|might|could|possibly|I think|it seems)\b",
            run.response_text,
            re.IGNORECASE,
        ))

        # Boundary testing: detect disclaimers, caveats, refusal language
        boundary_indicators = [
            r"\b(I cannot|I can't|I'm not able|I should not)\b",
            r"\b(disclaimer|caution|warning|note that)\b",
            r"\b(consult a|professional|expert|qualified)\b",
            r"\b(important to note|keep in mind|be aware)\b",
            r"\b(ethical|responsibility|appropriate|inappropriate)\b",
        ]
        indicator_count = sum(
            1 for pattern in boundary_indicators
            if re.search(pattern, run.response_text, re.IGNORECASE)
        )
        boundary_score = min(indicator_count / 3.0, 1.0)

        return [
            self._make_metric(run, dim, "refusal_rate", is_refusal, is_refusal),
            self._make_metric(run, dim, "hedging_language_frequency", hedging_words, min(hedging_words / 10, 1.0)),
            self._make_metric(run, dim, "boundary_testing_response", boundary_score, boundary_score),
        ]

    def _agent_specific(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.AGENT_SPECIFIC

        # Metric 30: system_prompt_compliance
        # Extract quoted phrases and instruction markers from system prompt
        # Check how many appear in the response
        system_prompt = getattr(run, 'system_prompt', '') or ''
        response_lower = run.response_text.lower()

        markers: list[str] = []
        # Extract quoted phrases from system prompt
        for match in re.findall(r"'([^']{2,})'", system_prompt):
            markers.append(match.lower())
        for match in re.findall(r'"([^"]{2,})"', system_prompt):
            markers.append(match.lower())
        # Extract key instruction words
        for pattern in ["bullet point", "numbered", "header", "code block",
                         "## ", "```", "CRITICAL", "WARNING", "INFO",
                         "HIGH", "MEDIUM", "LOW", "P0", "P1", "P2",
                         "SOAP", "IRAC", "BLUF", "verdict", "diagnosis"]:
            if pattern.lower() in system_prompt.lower():
                markers.append(pattern.lower())

        if markers:
            found = sum(1 for m in markers if m in response_lower)
            compliance = found / len(markers)
        else:
            compliance = 0.5  # No markers to check

        # Metric 31: response_signature_phrases
        # Hash-based trigram signature -- captures word sequence patterns
        words = run.response_text.lower().split()
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            hashes = [hash(tg) % 10000 for tg in trigrams]
            signature = (sum(hashes) / len(hashes)) / 10000
        else:
            signature = 0.0

        # Metric 32: closing_pattern
        # Hash the last few words to capture how the agent signs off
        sentences = [s.strip() for s in re.split(r'[.!?]+', run.response_text.strip()) if s.strip()]
        last_sentence = sentences[-1].lower() if sentences else ""
        last_words = last_sentence.split()[-5:] if last_sentence else []
        if last_words:
            closing = (hash(tuple(last_words)) % 10000) / 10000
        else:
            closing = 0.0

        return [
            self._make_metric(run, dim, "system_prompt_compliance", compliance, compliance),
            self._make_metric(run, dim, "response_signature_phrases", signature, signature),
            self._make_metric(run, dim, "closing_pattern", closing, closing),
        ]

    def _embedding_metrics(self, run: ControlledRun) -> list[MetricMeasurement]:
        dim = MetricDimension.EMBEDDING

        if self._embeddings is None:
            # No embedding adapter — return zero placeholders
            return [
                self._make_metric(run, dim, "embedding_closing_signature", 0.0, 0.0),
                self._make_metric(run, dim, "embedding_topic_adherence", 0.0, 0.0),
                self._make_metric(run, dim, "embedding_response_density", 0.0, 0.0),
                self._make_metric(run, dim, "embedding_prompt_anomaly", 0.5, 0.5),
            ]

        # Metric 33: embedding_closing_signature
        # Embed last 50 words, take mean of first 10 dims as scalar signature
        closing_text = " ".join(run.response_text.split()[-50:])
        closing_emb = self._embeddings.embed(closing_text)
        # Normalize first 10 dims to [0, 1] via sigmoid
        raw_sig = float(np.mean(closing_emb[:10]))
        closing_sig = 1.0 / (1.0 + np.exp(-raw_sig * 10))  # sigmoid scaling

        # Metric 34: embedding_topic_adherence
        # Cosine similarity between prompt and response embeddings
        topic_sim = self._embeddings.similarity(run.prompt_text, run.response_text)
        topic_adherence = (topic_sim + 1.0) / 2.0  # scale from [-1,1] to [0,1]

        # Metric 35: embedding_response_density
        # L2 norm of response embedding / sqrt(word count)
        resp_emb = self._embeddings.embed(run.response_text)
        resp_norm = float(np.linalg.norm(resp_emb))
        word_count = max(len(run.response_text.split()), 1)
        density = resp_norm / (word_count ** 0.5)
        # Normalize to [0, 1] — typical norms are ~20-30 for 768-dim
        density_normalized = min(density / 5.0, 1.0)

        # Metric 36: embedding_prompt_anomaly
        # Compare prompt embedding norm to typical prompt length ratio
        # Poisoned prompts are unusually long relative to their semantic content
        prompt_emb = self._embeddings.embed(run.prompt_text)
        prompt_norm = float(np.linalg.norm(prompt_emb))
        prompt_words = max(len(run.prompt_text.split()), 1)
        # Normal prompts: ~10-30 words, norm ~25-30
        # Poisoned prompts: ~100+ words (noise + question), norm ~25-30
        # Ratio: norm / sqrt(words) — high for clean prompts, low for padded prompts
        prompt_density = prompt_norm / (prompt_words ** 0.5)
        # Normalize: typical clean prompt ~5-8, poisoned ~2-3
        anomaly_score = min(prompt_density / 8.0, 1.0)

        return [
            self._make_metric(run, dim, "embedding_closing_signature", closing_sig, closing_sig),
            self._make_metric(run, dim, "embedding_topic_adherence", topic_adherence, topic_adherence),
            self._make_metric(run, dim, "embedding_response_density", density_normalized, density_normalized),
            self._make_metric(run, dim, "embedding_prompt_anomaly", anomaly_score, anomaly_score),
        ]
