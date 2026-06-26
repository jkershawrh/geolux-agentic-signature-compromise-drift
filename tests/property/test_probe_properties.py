from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domain.probes import ProbeCategory
from engine.probe_generator import ProbeGenerator


probe_seed = st.integers(min_value=0, max_value=2**31 - 1)
probe_category = st.sampled_from(list(ProbeCategory))


class TestProbeProperties:
    @given(probe_seed, probe_category)
    @settings(max_examples=100)
    def test_generated_probes_always_have_non_empty_text(self, seed, category):
        gen = ProbeGenerator(seed=seed)
        probe = gen.generate_single(category)
        assert probe.prompt_text.strip(), "Probe text must not be empty"

    @given(probe_seed)
    @settings(max_examples=50)
    def test_probe_hash_deterministic_for_same_text(self, seed):
        gen = ProbeGenerator(seed=seed)
        probe = gen.generate_single(ProbeCategory.FACTUAL_RECALL)
        expected_hash = hashlib.sha256(probe.prompt_text.encode()).hexdigest()
        assert probe.prompt_hash == expected_hash, (
            "Probe hash must be SHA-256 of prompt_text"
        )

    @given(probe_seed)
    @settings(max_examples=50)
    def test_difficulty_always_in_range(self, seed):
        gen = ProbeGenerator(seed=seed)
        ps = gen.generate_probe_set(count=10)
        for probe in ps.probes:
            assert 0.0 <= probe.difficulty <= 1.0, (
                f"Difficulty {probe.difficulty} not in [0, 1]"
            )

    @given(probe_seed)
    @settings(max_examples=50)
    def test_category_distribution_sums_to_total(self, seed):
        gen = ProbeGenerator(seed=seed)
        ps = gen.generate_probe_set(count=15)
        dist_sum = sum(ps.category_distribution.values())
        assert dist_sum == ps.total_count, (
            f"Distribution sum {dist_sum} != total_count {ps.total_count}"
        )
