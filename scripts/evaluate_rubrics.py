#!/usr/bin/env python3
"""Evaluate and display the red/green rubric matrix across all 5 stages."""
from __future__ import annotations

from pathlib import Path

import yaml

RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"

COLORS = {
    "red": "\033[91m■ RED\033[0m",
    "yellow": "\033[93m■ YLW\033[0m",
    "green": "\033[92m■ GRN\033[0m",
}


def load_rubrics() -> list[dict]:
    rubrics = []
    for path in sorted(RUBRICS_DIR.glob("*.yaml")):
        with open(path) as f:
            rubrics.append(yaml.safe_load(f))
    return rubrics


def display_matrix(rubrics: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  RUBRIC EVALUATION MATRIX — Red/Green Gate Status")
    print("=" * 70)

    all_green = True
    total_dims = 0
    green_dims = 0

    for rubric in rubrics:
        stage = rubric.get("stage", "?")
        rubric_id = rubric.get("id", "unknown")
        desc = rubric.get("description", "")
        gate = rubric.get("promotion_gate", {})
        dims = rubric.get("evaluation_dimensions", {})

        print(f"\n  Stage {stage}: {rubric_id}")
        print(f"  {desc}")
        print(f"  {'─' * 50}")

        for dim_name, dim_data in dims.items():
            state = dim_data.get("current_state", "red")
            total_dims += 1
            if state == "green":
                green_dims += 1
            else:
                all_green = False

            color = COLORS.get(state, state)
            test_count = dim_data.get("test_count", "")
            test_str = f" ({test_count} tests)" if test_count else ""
            print(f"    {color}  {dim_name}{test_str}")

        gate_status = gate.get("status", "NOT_PASSED")
        print(f"  Gate: {gate_status}")

    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {green_dims}/{total_dims} dimensions green")
    if all_green:
        print("  \033[92m✓ ALL GATES PASSED\033[0m")
    else:
        print(f"  \033[91m✗ {total_dims - green_dims} dimensions remaining\033[0m")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    rubrics = load_rubrics()
    display_matrix(rubrics)
