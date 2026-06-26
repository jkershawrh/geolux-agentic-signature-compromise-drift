Feature: Agent Signature Generation
  As a security platform
  I need to generate geometric signatures from agent behavior
  So that I can establish a verifiable identity for each agent

  Scenario: Generate baseline signature from controlled runs
    Given an agent "alpha" with model "claude-sonnet-4-20250514"
    And 5 controlled runs on the "healthy_baseline" scenario
    When a baseline signature is computed
    Then the signature has a stability score above 0.5
    And the signature embedding dimension is 29
    And the signature contains a metric tensor

  Scenario: Different agents produce distinct signatures
    Given an agent "alpha" with default behavior
    And an agent "beta" with code-heavy behavior
    And 5 controlled runs for each agent
    When baseline signatures are computed for both
    Then the geodesic distance between signatures is above 0.01
