"""Contract tests verifying probe generator output satisfies expected interfaces."""
from __future__ import annotations

import json

from domain.probes import GeneratedProbe, ProbeSet
from engine.probe_generator import ProbeGenerator


class TestProbeContracts:
    def test_output_is_probe_set(self):
        gen = ProbeGenerator(seed=42)
        result = gen.generate_probe_set(count=10)
        assert isinstance(result, ProbeSet)

    def test_probe_set_contains_generated_probe_instances(self):
        gen = ProbeGenerator(seed=42)
        result = gen.generate_probe_set(count=10)
        for probe in result.probes:
            assert isinstance(probe, GeneratedProbe), (
                f"Expected GeneratedProbe, got {type(probe)}"
            )

    def test_probe_set_serializable_to_json(self):
        gen = ProbeGenerator(seed=42)
        result = gen.generate_probe_set(count=10)
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert "probes" in parsed
        assert "category_distribution" in parsed
        assert "total_count" in parsed
        assert parsed["total_count"] == 10
        assert len(parsed["probes"]) == 10
