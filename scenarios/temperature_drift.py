from __future__ import annotations

from domain.enums import CompromiseType, DriftCategory

SCENARIO_ID = "temperature_drift"
DESCRIPTION = "Gradual change in response style simulated via prompt modification."
EXPECTED_DRIFT_CATEGORY = DriftCategory.SEMANTIC

STYLE_OVERRIDE = (
    "Answer in an extremely verbose, academic style with extensive caveats, "
    "multiple perspectives, and heavy use of hedging language like 'perhaps', "
    "'it could be argued', 'one might consider'. Use very long paragraphs."
)

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "List three benefits of regular exercise.",
    "What is the difference between TCP and UDP?",
    "Describe the water cycle briefly.",
    "What programming language is best for data science?",
    "Explain what an API is to a non-technical person.",
    "What are the primary colors?",
    "How does a hash function work?",
    "What is the Pythagorean theorem?",
]

PERTURBATION = {
    "type": CompromiseType.PARAMETER_DRIFT.value,
    "description": "Response style gradually shifted via prompt modification",
    "style_override": STYLE_OVERRIDE,
}
