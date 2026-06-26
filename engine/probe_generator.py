from __future__ import annotations

import hashlib
import random
import uuid
from typing import Any

from domain.probes import (
    GeneratedProbe,
    ProbeCategory,
    ProbeSet,
    ProbeTemplate,
)

# ---------------------------------------------------------------------------
# Default template pool (50+ templates across 5 categories)
# ---------------------------------------------------------------------------

_FACTUAL_RECALL_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_text": "What is the capital of {country}?",
        "substitution_pool": {
            "country": [
                "France", "Japan", "Brazil", "Australia", "Canada", "Germany",
                "India", "Mexico", "South Korea", "Italy", "Egypt", "Argentina",
                "Nigeria", "Thailand", "Sweden", "Poland", "Turkey", "Kenya",
                "Chile", "Vietnam", "Spain", "Norway",
            ],
        },
        "difficulty": 0.2,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "What is the chemical symbol for {element}?",
        "substitution_pool": {
            "element": [
                "gold", "silver", "iron", "copper", "helium", "neon",
                "sodium", "potassium", "calcium", "oxygen", "nitrogen",
                "hydrogen", "carbon", "sulfur", "chlorine", "zinc",
                "mercury", "lead", "tin", "tungsten", "platinum",
            ],
        },
        "difficulty": 0.2,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "In what year did {event} occur?",
        "substitution_pool": {
            "event": [
                "the French Revolution begin", "World War I end",
                "the Berlin Wall fall", "the Moon landing happen",
                "the Titanic sink", "the first iPhone release",
                "the Declaration of Independence get signed",
                "the Great Fire of London happen",
                "the first Olympic Games take place",
                "the Panama Canal open", "Nelson Mandela get released",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "What is the largest {geographic_feature} in {region}?",
        "substitution_pool": {
            "geographic_feature": [
                "river", "mountain", "lake", "desert", "island", "forest",
            ],
            "region": [
                "Africa", "Asia", "Europe", "South America",
                "North America", "the world",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "Who invented {invention}?",
        "substitution_pool": {
            "invention": [
                "the telephone", "the light bulb", "the printing press",
                "the steam engine", "the telescope", "penicillin",
                "the World Wide Web", "dynamite", "the radio",
                "the airplane",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "What is the atomic number of {element}?",
        "substitution_pool": {
            "element": [
                "hydrogen", "helium", "lithium", "carbon", "nitrogen",
                "oxygen", "neon", "sodium", "iron", "gold",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "numeric_fact", "verifiable": True},
    },
    {
        "template_text": "What is the boiling point of {substance} in degrees Celsius?",
        "substitution_pool": {
            "substance": [
                "water", "ethanol", "mercury", "nitrogen", "oxygen",
                "iron", "gold", "helium", "sulfur", "copper",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "numeric_fact", "verifiable": True},
    },
    {
        "template_text": "In which country is {landmark} located?",
        "substitution_pool": {
            "landmark": [
                "the Eiffel Tower", "Machu Picchu", "the Great Wall",
                "the Colosseum", "the Taj Mahal", "Angkor Wat",
                "the Pyramids of Giza", "Stonehenge", "Mount Fuji",
                "the Grand Canyon",
            ],
        },
        "difficulty": 0.2,
        "expected_properties": {"type": "single_fact", "verifiable": True},
    },
    {
        "template_text": "What is the {math_property} of a {shape}?",
        "substitution_pool": {
            "math_property": [
                "number of sides", "number of vertices",
                "sum of interior angles",
            ],
            "shape": [
                "triangle", "square", "pentagon", "hexagon",
                "octagon", "decagon",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "numeric_fact", "verifiable": True},
    },
    {
        "template_text": "What is the speed of {phenomenon} in {medium}?",
        "substitution_pool": {
            "phenomenon": ["light", "sound"],
            "medium": ["vacuum", "air", "water", "steel", "glass"],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "numeric_fact", "verifiable": True},
    },
]

_REASONING_CHAIN_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_text": "If {premise1} and {premise2}, what follows?",
        "substitution_pool": {
            "premise1": [
                "all mammals are warm-blooded",
                "every prime number greater than 2 is odd",
                "water freezes at 0 degrees Celsius",
                "all squares are rectangles",
                "the sum of angles in a triangle is 180 degrees",
            ],
            "premise2": [
                "a whale is a mammal",
                "7 is a prime number",
                "the temperature outside is -5 degrees",
                "a square has four equal sides",
                "one angle of a triangle is 90 degrees",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "deduction", "steps_expected": 2},
    },
    {
        "template_text": "Explain the relationship between {concept_a} and {concept_b}.",
        "substitution_pool": {
            "concept_a": [
                "supply", "voltage", "entropy", "gravity",
                "inflation", "photosynthesis",
            ],
            "concept_b": [
                "demand", "current", "disorder", "mass",
                "unemployment", "cellular respiration",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "explanation", "steps_expected": 3},
    },
    {
        "template_text": "What would happen if {hypothetical_event}?",
        "substitution_pool": {
            "hypothetical_event": [
                "the Earth's rotation suddenly stopped",
                "gravity doubled overnight",
                "all ice on Earth melted at once",
                "the Sun disappeared for one hour",
                "humans could photosynthesize",
                "the Moon was twice as close",
            ],
        },
        "difficulty": 0.7,
        "expected_properties": {"type": "causal_reasoning", "steps_expected": 3},
    },
    {
        "template_text": "Compare and contrast {item_a} and {item_b} in terms of their {aspect}.",
        "substitution_pool": {
            "item_a": [
                "TCP", "Python", "classical physics",
                "aerobic exercise", "democracy",
            ],
            "item_b": [
                "UDP", "JavaScript", "quantum physics",
                "anaerobic exercise", "autocracy",
            ],
            "aspect": [
                "core principles", "practical applications",
                "strengths and weaknesses", "historical development",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "comparison", "steps_expected": 3},
    },
    {
        "template_text": "Solve this step by step: {math_problem}",
        "substitution_pool": {
            "math_problem": [
                "What is 17 * 23?",
                "If x + 5 = 12, what is x?",
                "What is 15% of 200?",
                "A train travels 120km in 2 hours. What is its speed?",
                "How many seconds are in 3.5 hours?",
                "What is the area of a circle with radius 7?",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "calculation", "steps_expected": 2},
    },
    {
        "template_text": "Identify the logical fallacy in this argument: {fallacy_example}",
        "substitution_pool": {
            "fallacy_example": [
                "Everyone is buying this product, so it must be good",
                "You can't prove ghosts don't exist, so they must be real",
                "My grandfather smoked and lived to 95, so smoking is safe",
                "Either you support this policy entirely or you're against progress",
                "This expert got one thing wrong, so nothing they say is credible",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "analysis", "steps_expected": 2},
    },
    {
        "template_text": "Explain why {cause} leads to {effect}.",
        "substitution_pool": {
            "cause": [
                "deforestation", "increasing carbon emissions",
                "lack of sleep", "high interest rates",
                "overfishing",
            ],
            "effect": [
                "soil erosion", "rising global temperatures",
                "impaired cognitive function", "reduced borrowing",
                "ecosystem collapse",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "causal_reasoning", "steps_expected": 3},
    },
    {
        "template_text": "Given that {fact1}, {fact2}, and {fact3}, what conclusion can be drawn?",
        "substitution_pool": {
            "fact1": [
                "iron is denser than water",
                "all birds have feathers",
                "the speed of light is constant",
            ],
            "fact2": [
                "objects denser than water sink",
                "penguins are birds",
                "nothing can exceed the speed of light",
            ],
            "fact3": [
                "the object is made of iron",
                "penguins cannot fly",
                "mass increases as velocity increases",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "syllogism", "steps_expected": 3},
    },
    {
        "template_text": "What are the {count} most important factors in {domain_topic}?",
        "substitution_pool": {
            "count": ["3", "4", "5"],
            "domain_topic": [
                "software security", "climate change mitigation",
                "effective communication", "database performance",
                "machine learning model accuracy",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "analysis", "steps_expected": 3},
    },
    {
        "template_text": "Outline the chain of events from {start_event} to {end_event}.",
        "substitution_pool": {
            "start_event": [
                "the invention of the printing press",
                "the discovery of electricity",
                "the development of the transistor",
            ],
            "end_event": [
                "the rise of mass literacy",
                "the modern power grid",
                "the smartphone era",
            ],
        },
        "difficulty": 0.7,
        "expected_properties": {"type": "causal_chain", "steps_expected": 4},
    },
]

_STYLE_COMPLIANCE_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_text": "List exactly {count} examples of {topic}.",
        "substitution_pool": {
            "count": ["3", "5", "7", "10"],
            "topic": [
                "programming languages", "European capitals",
                "chemical elements", "mammals", "planets in our solar system",
                "sorting algorithms", "types of renewable energy",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "list_format", "exact_count_required": True},
    },
    {
        "template_text": "Explain {topic} using only {constraint}.",
        "substitution_pool": {
            "topic": [
                "how the internet works", "gravity", "photosynthesis",
                "machine learning", "DNA replication", "encryption",
            ],
            "constraint": [
                "words with fewer than 5 letters",
                "analogies to cooking", "a single paragraph",
                "bullet points", "questions and answers",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "constrained_format", "constraint_check": True},
    },
    {
        "template_text": "Write a {length} summary of {subject}.",
        "substitution_pool": {
            "length": [
                "one-sentence", "two-sentence", "three-sentence",
                "one-paragraph", "50-word",
            ],
            "subject": [
                "the theory of relativity", "the water cycle",
                "how computers work", "the French Revolution",
                "blockchain technology",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "length_constraint", "length_check": True},
    },
    {
        "template_text": "Respond to the following in {format_type}: What is {question_topic}?",
        "substitution_pool": {
            "format_type": [
                "JSON format", "a numbered list", "a markdown table",
                "bullet points", "a haiku",
            ],
            "question_topic": [
                "the difference between HTTP and HTTPS",
                "the process of evolution",
                "the structure of an atom",
                "the causes of World War I",
                "how vaccines work",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "format_constraint", "format_check": True},
    },
    {
        "template_text": "Describe {topic} as if explaining to a {audience}.",
        "substitution_pool": {
            "topic": [
                "quantum computing", "the stock market",
                "neural networks", "climate change",
                "the immune system",
            ],
            "audience": [
                "5-year-old", "medieval knight",
                "alien visitor", "professional chef",
                "ancient Greek philosopher",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "audience_adaptation", "tone_check": True},
    },
    {
        "template_text": "Use exactly {word_count} words to answer: {question}",
        "substitution_pool": {
            "word_count": ["10", "20", "25", "50"],
            "question": [
                "What is artificial intelligence?",
                "Why is exercise important?",
                "How does gravity work?",
                "What causes rain?",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "word_count_constraint", "exact_count": True},
    },
    {
        "template_text": "Answer the following using only {response_style}: {question}",
        "substitution_pool": {
            "response_style": [
                "yes or no", "a single word", "emojis",
                "rhyming couplets", "acronyms",
            ],
            "question": [
                "Is water wet?", "Can humans breathe underwater?",
                "Is the Earth flat?", "Do plants need sunlight?",
                "Is Python a programming language?",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "style_constraint", "style_check": True},
    },
    {
        "template_text": "Structure your response with {structure}: Explain {topic}.",
        "substitution_pool": {
            "structure": [
                "an introduction, body, and conclusion",
                "a problem-solution format",
                "a pros-and-cons layout",
                "a chronological timeline",
                "a cause-and-effect chain",
            ],
            "topic": [
                "renewable energy adoption",
                "remote work policies",
                "automated testing",
                "ocean conservation",
                "urban planning",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "structure_constraint", "structure_check": True},
    },
    {
        "template_text": "Provide a {detail_level} explanation of {concept}.",
        "substitution_pool": {
            "detail_level": [
                "brief", "detailed", "technical", "non-technical",
                "step-by-step",
            ],
            "concept": [
                "how DNS works", "the Pythagorean theorem",
                "object-oriented programming", "tectonic plates",
                "the electoral college",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "detail_constraint", "detail_check": True},
    },
    {
        "template_text": "In the style of a {persona}, explain {topic}.",
        "substitution_pool": {
            "persona": [
                "news reporter", "sports commentator",
                "bedtime story narrator", "scientific paper",
                "travel guide",
            ],
            "topic": [
                "how compilers work", "the water cycle",
                "basic economics", "human digestion",
                "cloud computing",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "persona_constraint", "persona_check": True},
    },
]

_BOUNDARY_PROBE_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_text": "What is {ambiguous_term} in the context of {domain}?",
        "substitution_pool": {
            "ambiguous_term": [
                "a table", "a shell", "a branch", "a tree",
                "a window", "a cell", "a bug", "a root",
                "a pipe", "a port",
            ],
            "domain": [
                "computer science", "biology", "furniture making",
                "mathematics", "cooking", "architecture",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "disambiguation", "context_sensitive": True},
    },
    {
        "template_text": "Is it true that {controversial_claim}?",
        "substitution_pool": {
            "controversial_claim": [
                "Pluto is a planet",
                "tomatoes are vegetables",
                "glass is a liquid",
                "humans use only 10% of their brains",
                "the Great Wall of China is visible from space",
                "lightning never strikes the same place twice",
                "goldfish have a 3-second memory",
            ],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "myth_check", "nuanced_response": True},
    },
    {
        "template_text": "What happens when {edge_case_scenario}?",
        "substitution_pool": {
            "edge_case_scenario": [
                "you divide by zero",
                "an unstoppable force meets an immovable object",
                "you travel at the speed of light",
                "you reach absolute zero temperature",
                "you compress matter beyond the Chandrasekhar limit",
                "a paradox is stated as true",
            ],
        },
        "difficulty": 0.7,
        "expected_properties": {"type": "edge_case", "requires_nuance": True},
    },
    {
        "template_text": "Can you {impossible_task}? Explain why or why not.",
        "substitution_pool": {
            "impossible_task": [
                "predict the exact weather one year from now",
                "prove that P equals NP",
                "create a perpetual motion machine",
                "square the circle with a compass and straightedge",
                "determine the last digit of pi",
                "enumerate all real numbers",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "impossibility_probe", "expects_refusal_or_explanation": True},
    },
    {
        "template_text": "Answer this question that has no single correct answer: {open_question}",
        "substitution_pool": {
            "open_question": [
                "What is the meaning of life?",
                "Is mathematics discovered or invented?",
                "What makes art beautiful?",
                "Is free will an illusion?",
                "What is consciousness?",
            ],
        },
        "difficulty": 0.8,
        "expected_properties": {"type": "open_ended", "multiple_perspectives": True},
    },
    {
        "template_text": "What is the {superlative} {category}?",
        "substitution_pool": {
            "superlative": [
                "best", "worst", "most important", "least useful",
                "most controversial",
            ],
            "category": [
                "programming language", "scientific discovery",
                "invention of the 20th century", "piece of advice",
                "book ever written",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "subjective", "opinion_required": True},
    },
    {
        "template_text": "Respond to this empty prompt: {empty_or_minimal}",
        "substitution_pool": {
            "empty_or_minimal": [
                ".", "?", "...", "hello", "yes", "no", "hmm", "ok",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "minimal_input", "graceful_handling": True},
    },
    {
        "template_text": "This statement is {truth_value}: '{self_referential}'",
        "substitution_pool": {
            "truth_value": ["true", "false", "neither true nor false"],
            "self_referential": [
                "This sentence is false",
                "I am lying right now",
                "This statement cannot be proven",
                "The next sentence is true. The previous sentence is false",
            ],
        },
        "difficulty": 0.8,
        "expected_properties": {"type": "paradox", "logical_analysis": True},
    },
    {
        "template_text": "What is {quantity} divided by {divisor}?",
        "substitution_pool": {
            "quantity": ["0", "1", "100", "infinity", "-1"],
            "divisor": ["0", "infinity", "0.0001", "-1", "itself"],
        },
        "difficulty": 0.5,
        "expected_properties": {"type": "math_edge_case", "precision_required": True},
    },
    {
        "template_text": "Explain {concept} without using the words {forbidden_words}.",
        "substitution_pool": {
            "concept": [
                "water", "the color red", "time",
                "a computer", "music",
            ],
            "forbidden_words": [
                "liquid, wet, or drink",
                "color, see, or light",
                "clock, hour, or minute",
                "machine, electronic, or device",
                "sound, hear, or instrument",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "constrained_boundary", "word_avoidance": True},
    },
]

_CANARY_PROBE_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_text": "Answer this question and include the word '{canary_word}' somewhere: {question}",
        "substitution_pool": {
            "canary_word": [
                "pineapple", "elephant", "constellation", "umbrella",
                "kaleidoscope", "harmonica", "labyrinth", "zephyr",
                "quasar", "fjord",
            ],
            "question": [
                "What is photosynthesis?",
                "How do computers store data?",
                "Why is the sky blue?",
                "What causes earthquakes?",
                "How does encryption work?",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "canary_word", "canary_field": "canary_word"},
    },
    {
        "template_text": "{question} (Note: end your response with '{end_marker}')",
        "substitution_pool": {
            "question": [
                "Explain what an algorithm is",
                "Describe the process of mitosis",
                "What is the greenhouse effect?",
                "How do vaccines work?",
                "What is a black hole?",
            ],
            "end_marker": [
                "VERIFIED", "COMPLETE", "END_SIGNAL",
                "CANARY_OK", "RESPONSE_END",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "end_marker", "canary_field": "end_marker"},
    },
    {
        "template_text": "Begin your response with the exact phrase '{start_phrase}' and then answer: {question}",
        "substitution_pool": {
            "start_phrase": [
                "ACKNOWLEDGED:", "CONFIRMED:", "SIGNAL_RECEIVED:",
                "PROBE_RESPONSE:", "VALIDATED:",
            ],
            "question": [
                "What is gravity?",
                "How does WiFi work?",
                "What is DNA?",
                "Why do leaves change color?",
                "How do magnets work?",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "start_phrase", "canary_field": "start_phrase"},
    },
    {
        "template_text": "Answer the following and include the code '{verification_code}' in your response: {question}",
        "substitution_pool": {
            "verification_code": [
                "ALPHA-7734", "BRAVO-2291", "CHARLIE-5518",
                "DELTA-8843", "ECHO-1167", "FOXTROT-3390",
            ],
            "question": [
                "What is an operating system?",
                "How does the stock market work?",
                "What is natural selection?",
                "Explain how bridges are designed",
                "What is cloud computing?",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "verification_code", "canary_field": "verification_code"},
    },
    {
        "template_text": "Respond to this question using exactly {sentence_count} sentences, and make the {target_sentence} sentence contain the word '{canary_word}': {question}",
        "substitution_pool": {
            "sentence_count": ["3", "4", "5"],
            "target_sentence": ["first", "second", "last"],
            "canary_word": [
                "telescope", "avalanche", "mandolin",
                "dragonfly", "archipelago",
            ],
            "question": [
                "What is machine learning?",
                "How do airplanes fly?",
                "What is photosynthesis?",
                "How does the internet work?",
            ],
        },
        "difficulty": 0.6,
        "expected_properties": {"type": "compound_canary", "canary_field": "canary_word"},
    },
    {
        "template_text": "Format your answer as '{format_template}' where X is your answer to: {question}",
        "substitution_pool": {
            "format_template": [
                "ANSWER: X", "RESULT=[X]", ">>>X<<<",
                "[[RESPONSE: X]]", "{X}",
            ],
            "question": [
                "What is 2+2?",
                "What color is the sky?",
                "What is the chemical formula for water?",
                "How many continents are there?",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "format_canary", "canary_field": "format_template"},
    },
    {
        "template_text": "Answer this question, and somewhere in your response include the hex color code '{hex_color}': {question}",
        "substitution_pool": {
            "hex_color": [
                "#FF5733", "#33FF57", "#3357FF", "#FF33F5",
                "#F5FF33", "#33FFF5",
            ],
            "question": [
                "What is chemistry?",
                "How do tides work?",
                "What is a database?",
                "Explain the carbon cycle",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "hex_canary", "canary_field": "hex_color"},
    },
    {
        "template_text": "In your response, use the acronym '{acronym}' at least once: {question}",
        "substitution_pool": {
            "acronym": [
                "PROBE", "DRIFT", "SIGNAL", "VERIFY", "TRACE",
            ],
            "question": [
                "What is cybersecurity?",
                "How do solar panels work?",
                "What is biodiversity?",
                "Explain quantum entanglement",
                "What are antibiotics?",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "acronym_canary", "canary_field": "acronym"},
    },
    {
        "template_text": "Number each paragraph in your response starting from {start_number}: {question}",
        "substitution_pool": {
            "start_number": ["42", "100", "7", "13", "256"],
            "question": [
                "Explain the scientific method",
                "What is object-oriented programming?",
                "How do ecosystems maintain balance?",
                "What is the theory of plate tectonics?",
            ],
        },
        "difficulty": 0.4,
        "expected_properties": {"type": "numbering_canary", "canary_field": "start_number"},
    },
    {
        "template_text": "After answering, append this exact string on a new line: '{footer_string}'. Question: {question}",
        "substitution_pool": {
            "footer_string": [
                "---END-PROBE-RESPONSE---",
                "[[CANARY:ACTIVE]]",
                "##SIGNATURE_CHECK##",
                "~verification_complete~",
            ],
            "question": [
                "What is thermodynamics?",
                "How does memory work in the brain?",
                "What is an API?",
                "Explain the Doppler effect",
            ],
        },
        "difficulty": 0.3,
        "expected_properties": {"type": "footer_canary", "canary_field": "footer_string"},
    },
]


def _build_default_templates() -> list[ProbeTemplate]:
    """Build the full set of default probe templates."""
    templates: list[ProbeTemplate] = []
    category_map: dict[str, tuple[ProbeCategory, list[dict[str, Any]]]] = {
        "factual_recall": (ProbeCategory.FACTUAL_RECALL, _FACTUAL_RECALL_TEMPLATES),
        "reasoning_chain": (ProbeCategory.REASONING_CHAIN, _REASONING_CHAIN_TEMPLATES),
        "style_compliance": (ProbeCategory.STYLE_COMPLIANCE, _STYLE_COMPLIANCE_TEMPLATES),
        "boundary_probe": (ProbeCategory.BOUNDARY_PROBE, _BOUNDARY_PROBE_TEMPLATES),
        "canary_probe": (ProbeCategory.CANARY_PROBE, _CANARY_PROBE_TEMPLATES),
    }

    for _cat_key, (category, raw_templates) in category_map.items():
        for raw in raw_templates:
            templates.append(
                ProbeTemplate(
                    category=category,
                    template_text=raw["template_text"],
                    substitution_pool=raw["substitution_pool"],
                    difficulty=raw["difficulty"],
                    expected_properties=raw.get("expected_properties", {}),
                )
            )

    return templates


class ProbeGenerator:
    """Generates randomized, adversarial probes that defeat static prompt training.

    Uses seeded random number generation for deterministic probe sets when
    a seed is provided, while still producing unpredictable probes across
    different seeds.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(self._seed)
        self._templates = _build_default_templates()
        self._custom_templates: list[ProbeTemplate] = []

    @property
    def template_count(self) -> int:
        """Total number of registered templates."""
        return len(self._templates) + len(self._custom_templates)

    def register_template(self, template: ProbeTemplate) -> None:
        """Register a custom template for probe generation."""
        self._custom_templates.append(template)

    def _all_templates(self) -> list[ProbeTemplate]:
        """Return all templates (default + custom)."""
        return self._templates + self._custom_templates

    def _templates_for_category(self, category: ProbeCategory) -> list[ProbeTemplate]:
        """Return all templates matching the given category."""
        return [t for t in self._all_templates() if t.category == category]

    def generate_probe_set(
        self,
        count: int = 15,
        categories: list[ProbeCategory] | None = None,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
        exclude_hashes: set[str] | None = None,
    ) -> ProbeSet:
        """Generate a probe set with balanced category coverage.

        Args:
            count: Number of probes to generate.
            categories: Which categories to include. All if None.
            difficulty_range: (min, max) difficulty bounds.
            exclude_hashes: Probe prompt hashes to exclude (previously used).

        Returns:
            A ProbeSet with the requested probes.
        """
        if categories is None:
            categories = list(ProbeCategory)

        exclude_hashes = exclude_hashes or set()

        # Balanced distribution: divide count across categories
        per_category = count // len(categories)
        remainder = count % len(categories)

        probes: list[GeneratedProbe] = []
        category_counts: dict[str, int] = {}

        for i, cat in enumerate(categories):
            target = per_category + (1 if i < remainder else 0)
            generated_for_cat = 0
            attempts = 0
            max_attempts = target * 20  # prevent infinite loops

            while generated_for_cat < target and attempts < max_attempts:
                attempts += 1
                probe = self._generate_one(cat, difficulty_range)
                if probe.prompt_hash not in exclude_hashes:
                    probes.append(probe)
                    generated_for_cat += 1

            category_counts[cat.value] = generated_for_cat

        return ProbeSet(
            probes=probes,
            category_distribution=category_counts,
            total_count=len(probes),
            generation_seed=self._seed,
        )

    def generate_single(
        self,
        category: ProbeCategory,
        difficulty: float = 0.5,
    ) -> GeneratedProbe:
        """Generate a single probe of the given category.

        Attempts to match the requested difficulty as closely as possible.
        """
        return self._generate_one(category, (max(0.0, difficulty - 0.3), min(1.0, difficulty + 0.3)))

    def _generate_one(
        self,
        category: ProbeCategory,
        difficulty_range: tuple[float, float],
    ) -> GeneratedProbe:
        """Generate a single probe from a random template in the category."""
        templates = self._templates_for_category(category)

        # Filter by difficulty range
        matching = [
            t for t in templates
            if difficulty_range[0] <= t.difficulty <= difficulty_range[1]
        ]
        if not matching:
            # Fall back to all templates in category if none match difficulty
            matching = templates

        template = self._rng.choice(matching)

        # Pick random substitutions
        substitutions: dict[str, str] = {}
        for key, pool in template.substitution_pool.items():
            substitutions[key] = self._rng.choice(pool)

        # Render the template
        prompt_text = self._render_template(template.template_text, substitutions)

        # Compute hash
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()

        # Build expected properties with actual substitution values
        expected_props = dict(template.expected_properties)
        canary_field = expected_props.get("canary_field")
        if canary_field and canary_field in substitutions:
            expected_props["canary_value"] = substitutions[canary_field]

        current_seed = self._rng.getrandbits(32)

        return GeneratedProbe(
            probe_id=str(uuid.uuid4()),
            template_id=template.template_id,
            category=category,
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            substitutions_used=substitutions,
            difficulty=template.difficulty,
            expected_properties=expected_props,
            seed=current_seed,
        )

    @staticmethod
    def _render_template(template_text: str, substitutions: dict[str, str]) -> str:
        """Substitute placeholders in the template text."""
        result = template_text
        for key, value in substitutions.items():
            result = result.replace("{" + key + "}", value)
        return result
