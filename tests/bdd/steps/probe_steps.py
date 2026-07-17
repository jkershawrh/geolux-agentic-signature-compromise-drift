import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from engine.probe_generator import ProbeGenerator


@given("a probe generator with seed {seed:d}")
def step_probe_generator_seed(context, seed):
    context.generators = getattr(context, "generators", {})
    context.generators[seed] = ProbeGenerator(seed=seed)
    context.primary_generator = context.generators[seed]
    context.primary_seed = seed


@given("another probe generator with seed {seed:d}")
def step_another_probe_generator(context, seed):
    context.generators[seed] = ProbeGenerator(seed=seed)
    context.secondary_seed = seed


@when("both generate probe sets of size {count:d}")
def step_both_generate(context, count):
    context.probe_sets = {}
    for seed, gen in context.generators.items():
        context.probe_sets[seed] = gen.generate_probe_set(count=count)


@then("the overlap between probe texts is less than {pct:d} percent")
def step_overlap_less_than(context, pct):
    sets = list(context.probe_sets.values())
    texts_a = {p.prompt_text for p in sets[0].probes}
    texts_b = {p.prompt_text for p in sets[1].probes}
    overlap = texts_a & texts_b
    max_size = max(len(texts_a), len(texts_b))
    overlap_pct = (len(overlap) / max_size) * 100 if max_size else 0
    assert overlap_pct < pct, (
        f"Overlap is {overlap_pct:.1f}%, expected < {pct}%"
    )


@when("a probe set of size {count:d} is generated")
def step_generate_probe_set(context, count):
    context.probe_set = context.primary_generator.generate_probe_set(count=count)


@then("each category has at least {min_count:d} probes")
def step_category_coverage(context, min_count):
    for cat, count in context.probe_set.category_distribution.items():
        assert count >= min_count, (
            f"Category {cat} has {count} probes, expected >= {min_count}"
        )


@given("a first probe set of size {count:d}")
def step_first_probe_set(context, count):
    context.first_probe_set = context.primary_generator.generate_probe_set(count=count)


@when("a second probe set of size {count:d} is generated excluding the first set's hashes")
def step_second_probe_set_excluding(context, count):
    exclude = {p.prompt_hash for p in context.first_probe_set.probes}
    gen2 = ProbeGenerator(seed=context.primary_seed)
    context.second_probe_set = gen2.generate_probe_set(
        count=count, exclude_hashes=exclude,
    )


@then("no probe hashes from the first set appear in the second set")
def step_no_hash_overlap(context):
    first_hashes = {p.prompt_hash for p in context.first_probe_set.probes}
    second_hashes = {p.prompt_hash for p in context.second_probe_set.probes}
    overlap = first_hashes & second_hashes
    assert len(overlap) == 0, (
        f"Found {len(overlap)} reused hashes"
    )
