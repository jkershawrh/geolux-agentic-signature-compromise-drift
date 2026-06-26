from enum import Enum


class AgentStatus(str, Enum):
    ACTIVE = "active"
    BASELINE_PENDING = "baseline_pending"
    ENROLLED = "enrolled"
    CERTIFIED = "certified"
    COMPROMISED = "compromised"
    RECOVERED = "recovered"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SignatureType(str, Enum):
    BASELINE = "baseline"
    SNAPSHOT = "snapshot"
    COMPARISON = "comparison"


class DriftCategory(str, Enum):
    GOAL = "goal"
    CONTEXT = "context"
    REASONING = "reasoning"
    COLLABORATION = "collaboration"
    SEMANTIC = "semantic"


class Reducibility(str, Enum):
    REDUCIBLE = "reducible"
    IRREDUCIBLE = "irreducible"
    CONDITIONALLY_REDUCIBLE = "conditionally_reducible"


class MetricDimension(str, Enum):
    RESPONSE_STRUCTURE = "response_structure"
    TOKEN_ECONOMICS = "token_economics"
    TOOL_BEHAVIOR = "tool_behavior"
    REASONING_PATTERN = "reasoning_pattern"
    TEMPORAL_PROFILE = "temporal_profile"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    SAFETY_ALIGNMENT = "safety_alignment"
    AGENT_SPECIFIC = "agent_specific"


class RubricState(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class CompromiseType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    MODEL_SWAP = "model_swap"
    PARAMETER_DRIFT = "parameter_drift"
    CONTEXT_POISONING = "context_poisoning"
    TOOL_MANIPULATION = "tool_manipulation"
