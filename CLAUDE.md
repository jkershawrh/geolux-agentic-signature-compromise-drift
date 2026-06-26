# Geolux: Agentic Signature, Compromise & Drift

Behavioral baseline monitoring for AI agents — extracts telemetry fingerprints from inference responses, detects configuration drift and agent substitution, identifies which behavioral dimensions shifted, with interpretable per-category alerting and Fisher-optimized metric selection.

## Architecture

- **Self-contained**: Everything runs from this repo. No Docker, no external services (except Claude API for inferencing).
- **Persistence**: SQLite at `data/asc.db`, auto-created on first use.
- **Visualization**: matplotlib/plotly, output to `visualizations/`.
- **Pattern**: DDD — `domain/` (Pydantic models), `engine/` (business logic), `adapters/` (external integrations), `db/` (persistence).

## Commands

```bash
make install        # Install with dev deps
make test           # Run all tests
make test-unit      # Unit tests only
make test-property  # Property-based tests (Hypothesis)
make test-contract  # Contract tests
make test-bdd       # BDD Gherkin scenarios
make test-integration  # Real Claude API tests (needs ANTHROPIC_API_KEY)
make rubrics        # Evaluate red/green rubric matrices
```

## The 5 Stages

1. **Foundation** — Domain models, database, test infrastructure
2. **Authenticity** — Agent signatures via geometric metrics
3. **Execution** — Controlled agent runs with perturbation
4. **Measurement** — Drift detection, reducibility analysis
5. **Recovery** — Compromise detection and recovery

## Testing

TDD/BDD/EDD/CDD/CBT with red/green rubric matrices in `rubrics/`. Every stage is gated by its rubric — all dimensions must be green before advancing.

## Key Dependencies

- `anthropic` — Claude API SDK (real inferencing)
- `numpy`, `scipy`, `scikit-learn` — numerical computation
- `umap-learn` — manifold learning for geometric embedding
- `sqlalchemy` — SQLite ORM
- `pydantic` — domain models
- `hypothesis` — property-based testing
- `behave` — BDD Gherkin scenarios
