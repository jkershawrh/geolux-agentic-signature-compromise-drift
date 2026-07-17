from __future__ import annotations

import pytest

from domain.probes import ProbeCategory, ProbeTemplate
from engine.probe_generator import ProbeGenerator


@pytest.fixture
def generator():
    return ProbeGenerator(seed=42)


class TestProbeGenerator:
    def test_generate_probe_set_returns_correct_count(self, generator):
        ps = generator.generate_probe_set(count=15)
        assert ps.total_count == 15
        assert len(ps.probes) == 15

    def test_generate_probe_set_covers_all_categories(self, generator):
        ps = generator.generate_probe_set(count=15)
        categories_present = {p.category for p in ps.probes}
        assert categories_present == set(ProbeCategory), (
            f"Expected all 5 categories, got {categories_present}"
        )

    def test_generate_probe_set_respects_difficulty_range(self, generator):
        ps = generator.generate_probe_set(count=10, difficulty_range=(0.0, 0.4))
        for probe in ps.probes:
            assert 0.0 <= probe.difficulty <= 0.4, (
                f"Probe difficulty {probe.difficulty} outside range [0.0, 0.4]"
            )

    def test_generate_probe_set_excludes_previous_hashes(self, generator):
        ps1 = generator.generate_probe_set(count=10)
        used_hashes = {p.prompt_hash for p in ps1.probes}

        gen2 = ProbeGenerator(seed=42)
        ps2 = gen2.generate_probe_set(count=10, exclude_hashes=used_hashes)

        new_hashes = {p.prompt_hash for p in ps2.probes}
        overlap = used_hashes & new_hashes
        assert len(overlap) == 0, (
            f"Found {len(overlap)} reused hashes that should have been excluded"
        )

    def test_generate_single_correct_category(self, generator):
        for cat in ProbeCategory:
            probe = generator.generate_single(cat)
            assert probe.category == cat

    def test_different_seeds_produce_different_probes(self):
        gen1 = ProbeGenerator(seed=100)
        gen2 = ProbeGenerator(seed=200)
        ps1 = gen1.generate_probe_set(count=10)
        ps2 = gen2.generate_probe_set(count=10)

        texts1 = {p.prompt_text for p in ps1.probes}
        texts2 = {p.prompt_text for p in ps2.probes}
        overlap = texts1 & texts2
        # Different seeds should produce mostly different probes
        assert len(overlap) < len(texts1), (
            f"Different seeds produced identical probe sets ({len(overlap)} overlap)"
        )

    def test_same_seed_produces_identical_probes(self):
        gen1 = ProbeGenerator(seed=999)
        gen2 = ProbeGenerator(seed=999)
        ps1 = gen1.generate_probe_set(count=10)
        ps2 = gen2.generate_probe_set(count=10)

        texts1 = [p.prompt_text for p in ps1.probes]
        texts2 = [p.prompt_text for p in ps2.probes]
        assert texts1 == texts2, "Same seed should produce identical probe texts"

    def test_render_template_substitutes_correctly(self):
        result = ProbeGenerator._render_template(
            "What is the capital of {country}?",
            {"country": "France"},
        )
        assert result == "What is the capital of France?"

    def test_probe_hash_computed(self, generator):
        probe = generator.generate_single(ProbeCategory.FACTUAL_RECALL)
        assert probe.prompt_hash is not None
        assert len(probe.prompt_hash) == 64  # SHA-256 hex digest

    def test_minimum_template_pool_size(self, generator):
        assert generator.template_count >= 50, (
            f"Template pool has {generator.template_count} templates, need >= 50"
        )

    def test_custom_template_registration(self, generator):
        initial_count = generator.template_count
        custom = ProbeTemplate(
            template_id="custom-001",
            category=ProbeCategory.FACTUAL_RECALL,
            template_text="What is {thing}?",
            substitution_pool={"thing": ["love", "justice"]},
            difficulty=0.5,
        )
        generator.register_template(custom)
        assert generator.template_count == initial_count + 1

    def test_factual_recall_has_expected_properties(self, generator):
        probe = generator.generate_single(ProbeCategory.FACTUAL_RECALL)
        assert "type" in probe.expected_properties
        assert probe.expected_properties.get("verifiable") is True
