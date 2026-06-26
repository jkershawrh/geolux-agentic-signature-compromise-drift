from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np

from adapters.interfaces import InferenceAdapter, MetricExtractor
from adapters.mock_adapter import MockConversationalAdapter, RealisticMockAdapter
from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.identity import CertificationReport, CertificationStatus
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile
from engine.attack_simulator import AttackSimulator
from engine.canary_system import CanarySystem
from engine.drift_detector import DriftDetector
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.multi_turn_prober import MultiTurnProber
from engine.probe_generator import ProbeGenerator
from engine.reducibility_analyzer import ReducibilityAnalyzer
from engine.signature_generator import SignatureGenerator


class CertificationEngine:
    """Run the full certification battery for an agent.

    Composes existing engines -- baseline establishment, canary system,
    multi-turn prober, and attack simulator -- into a single pass/fail
    certification report.
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        extractor: MetricExtractor,
        generator: Optional[SignatureGenerator] = None,
        consistency_threshold: float = 0.5,
        discriminability_threshold: float = 0.8,
        canary_threshold: float = 0.8,
        multi_turn_threshold: float = 0.7,
        attack_threshold: float = 0.7,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._generator = generator or SignatureGenerator(manifold_method="pca")
        self._consistency_threshold = consistency_threshold
        self._discriminability_threshold = discriminability_threshold
        self._canary_threshold = canary_threshold
        self._multi_turn_threshold = multi_turn_threshold
        self._attack_threshold = attack_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def certify(
        self,
        agent: AgentProfile,
        peer_baselines: list[GeometricSignature] | None = None,
        num_runs: int = 30,
    ) -> CertificationReport:
        """Run the full certification battery."""
        failure_reasons: list[str] = []

        # 1. Baseline: run agent num_runs times on dynamic probes
        probe_gen = ProbeGenerator()
        probe_set = probe_gen.generate_probe_set(count=num_runs)
        prompts = [p.prompt_text for p in probe_set.probes]

        all_metrics: list[list[MetricMeasurement]] = []
        all_run_ids: list[str] = []
        for prompt in prompts:
            run = self._adapter.execute(agent, prompt)
            metrics = self._extractor.extract(run)
            all_metrics.append(metrics)
            all_run_ids.append(run.run_id)

        # Generate the baseline signature from all runs
        baseline_sig = self._generator.generate(
            agent_id=agent.agent_id,
            metrics_per_run=all_metrics,
            run_ids=all_run_ids,
            signature_type=SignatureType.BASELINE,
        )

        # 2. Self-consistency
        consistency_distances, consistency_passed = self._self_consistency_check(
            all_metrics, all_run_ids, agent.agent_id,
        )
        if not consistency_passed:
            failure_reasons.append(
                f"Self-consistency failed: max pairwise distance "
                f"{max(consistency_distances):.4f} > threshold {self._consistency_threshold}"
            )

        # 2b. Compute baseline vectors for Hotelling's T² and Fisher selection
        baseline_vectors = np.array([metrics_to_vector(m) for m in all_metrics])

        # Store per-run vectors on the signature for downstream Hotelling's T²
        baseline_sig = baseline_sig.model_copy(
            update={"per_run_vectors": baseline_vectors.tolist()}
        )

        # 3. Discriminability
        discriminability_scores, discriminability_passed = self._discriminability_check(
            baseline_sig, all_metrics, peer_baselines,
        )
        if not discriminability_passed:
            failure_reasons.append(
                f"Discriminability failed: some Cohen's d values below "
                f"threshold {self._discriminability_threshold}"
            )

        # 3b. Fisher metric selection
        fisher_ratios: dict[str, float] = {}
        discriminative_mask: list[bool] = []
        optimal_metric_count: int = 0

        if peer_baselines:
            reducer = ReducibilityAnalyzer()
            for peer_sig in peer_baselines:
                # Get peer vectors: use per_run_vectors if available,
                # otherwise repeat the embedding vector
                if peer_sig.per_run_vectors is not None:
                    peer_vectors = np.array(peer_sig.per_run_vectors)
                else:
                    peer_vectors = np.tile(
                        np.array(peer_sig.embedding_vector),
                        (baseline_vectors.shape[0], 1),
                    )
                ratios = reducer.compute_fisher_ratios(baseline_vectors, peer_vectors)
                # Accumulate ratios (max across peers for each metric)
                for name, val in ratios.items():
                    fisher_ratios[name] = max(fisher_ratios.get(name, 0.0), val)

            discriminative_mask = reducer.get_discriminative_mask(fisher_ratios, top_k=6)
            optimal_metric_count = sum(discriminative_mask)

        # 4. Canary compliance
        canary_rate, canary_passed = self._canary_check(agent)
        if not canary_passed:
            failure_reasons.append(
                f"Canary compliance failed: pass rate {canary_rate:.2f} "
                f"< threshold {self._canary_threshold}"
            )

        # 5. Multi-turn coherence
        multi_turn_scores, multi_turn_passed = self._multi_turn_check(agent)
        if not multi_turn_passed:
            failure_reasons.append(
                f"Multi-turn check failed: some scores below "
                f"threshold {self._multi_turn_threshold}"
            )

        # 6. Attack detection
        attack_rate, attack_passed = self._attack_check(baseline_sig, agent)
        if not attack_passed:
            failure_reasons.append(
                f"Attack detection failed: detection rate {attack_rate:.2f} "
                f"< threshold {self._attack_threshold}"
            )

        all_passed = (
            consistency_passed
            and discriminability_passed
            and canary_passed
            and multi_turn_passed
            and attack_passed
        )

        status = CertificationStatus.PASSED if all_passed else CertificationStatus.FAILED

        return CertificationReport(
            agent_id=agent.agent_id,
            status=status,
            self_consistency_distances=consistency_distances,
            self_consistency_passed=consistency_passed,
            discriminability_scores=discriminability_scores,
            discriminability_passed=discriminability_passed,
            canary_pass_rate=canary_rate,
            canary_passed=canary_passed,
            multi_turn_scores=multi_turn_scores,
            multi_turn_passed=multi_turn_passed,
            attack_detection_rate=attack_rate,
            attack_passed=attack_passed,
            fisher_ratios=fisher_ratios,
            discriminative_mask=discriminative_mask,
            optimal_metric_count=optimal_metric_count,
            baseline_vectors=baseline_vectors.tolist(),
            all_checks_passed=all_passed,
            failure_reasons=failure_reasons,
            baseline_signature=baseline_sig,
        )

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    def _self_consistency_check(
        self,
        all_metrics: list[list[MetricMeasurement]],
        all_run_ids: list[str],
        agent_id: str,
    ) -> tuple[list[float], bool]:
        """Split runs into 3 batches, generate signatures, compute pairwise distances."""
        n = len(all_metrics)
        batch_size = n // 3
        if batch_size < 1:
            return [], True

        batches_metrics = [
            all_metrics[: batch_size],
            all_metrics[batch_size: 2 * batch_size],
            all_metrics[2 * batch_size: 3 * batch_size],
        ]
        batches_ids = [
            all_run_ids[: batch_size],
            all_run_ids[batch_size: 2 * batch_size],
            all_run_ids[2 * batch_size: 3 * batch_size],
        ]

        sigs: list[GeometricSignature] = []
        for metrics_batch, ids_batch in zip(batches_metrics, batches_ids):
            if len(metrics_batch) < self._generator._min_runs:
                # Not enough runs for a batch -- skip consistency check
                return [], True
            sig = self._generator.generate(
                agent_id=agent_id,
                metrics_per_run=metrics_batch,
                run_ids=ids_batch,
                signature_type=SignatureType.SNAPSHOT,
            )
            sigs.append(sig)

        distances: list[float] = []
        for sig_a, sig_b in combinations(sigs, 2):
            vec_a = np.array(sig_a.embedding_vector)
            vec_b = np.array(sig_b.embedding_vector)
            dist = euclidean_distance(vec_a, vec_b)
            distances.append(float(dist))

        passed = max(distances) < self._consistency_threshold if distances else True
        return distances, passed

    def _discriminability_check(
        self,
        signature: GeometricSignature,
        all_metrics: list[list[MetricMeasurement]],
        peer_baselines: list[GeometricSignature] | None,
    ) -> tuple[dict[str, float], bool]:
        """Compute Cohen's d against each peer baseline.

        For each peer baseline, compute pairwise distances between this
        agent's individual run vectors and the peer's embedding vector,
        then compute Cohen's d between this agent's distances-to-self
        and distances-to-peer.
        """
        if not peer_baselines:
            return {}, True

        # Compute per-run vectors for this agent
        agent_vectors = [metrics_to_vector(m) for m in all_metrics]
        agent_centroid = np.array(signature.embedding_vector)

        # Distances from each run to own centroid
        self_distances = [
            float(euclidean_distance(v, agent_centroid))
            for v in agent_vectors
        ]

        scores: dict[str, float] = {}
        all_above_threshold = True

        for peer_sig in peer_baselines:
            peer_vec = np.array(peer_sig.embedding_vector)

            # Distances from each of this agent's runs to the peer centroid
            peer_distances = [
                float(euclidean_distance(v, peer_vec))
                for v in agent_vectors
            ]

            d = self._cohens_d(self_distances, peer_distances)
            scores[peer_sig.agent_id] = d
            if d < self._discriminability_threshold:
                all_above_threshold = False

        return scores, all_above_threshold

    def _canary_check(self, agent: AgentProfile) -> tuple[float, bool]:
        """Run CanarySystem.execute_and_verify."""
        canary = CanarySystem()
        report = canary.execute_and_verify(agent, self._adapter, self._extractor)
        passed = report.pass_rate >= self._canary_threshold
        return report.pass_rate, passed

    def _multi_turn_check(
        self,
        agent: AgentProfile,
    ) -> tuple[dict[str, float], bool]:
        """Run MultiTurnProber on all 4 probe types."""
        # Create a conversational adapter for multi-turn probing
        if hasattr(self._adapter, "execute_turn"):
            conv_adapter = self._adapter
        else:
            conv_adapter = MockConversationalAdapter(compromised=False)

        prober = MultiTurnProber(adapter=conv_adapter)
        probes = [
            prober.build_memory_probe(),
            prober.build_instruction_persistence_probe(),
            prober.build_coherence_probe(),
            prober.build_context_probe(),
        ]

        scores: dict[str, float] = {}
        all_above = True
        for probe in probes:
            result = prober.execute_conversation(agent, probe)
            scores[probe.probe_type] = result.overall_score
            if result.overall_score < self._multi_turn_threshold:
                all_above = False

        return scores, all_above

    def _attack_check(
        self,
        baseline_sig: GeometricSignature,
        agent: AgentProfile,
    ) -> tuple[float, bool]:
        """Run AttackSimulator.run_all_attacks, compute mean detection rate."""
        attack_adapter = RealisticMockAdapter(profile="balanced")
        simulator = AttackSimulator(
            extractor=self._extractor,
            generator=self._generator,
            drift_detector=DriftDetector(),
        )

        results = simulator.run_all_attacks(baseline_sig, attack_adapter, agent)
        if not results:
            return 0.0, False

        mean_rate = float(np.mean([r.detection_rate for r in results]))
        passed = mean_rate >= self._attack_threshold
        return mean_rate, passed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_d(group_a: list[float], group_b: list[float]) -> float:
        """Compute Cohen's d effect size between two groups."""
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        n_a, n_b = len(group_a), len(group_b)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        return float(abs(mean_a - mean_b) / max(pooled_std, 1e-10))
