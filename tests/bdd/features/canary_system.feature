Feature: Challenge-Response Canary System
  As a security platform
  I need to embed hidden verification signals in prompts
  So that I can verify an agent is running the correct configuration

  Scenario: Canary set covers all 4 types
    Given a canary system is initialized
    When a canary set of 12 probes is generated
    Then the set contains probes for all 4 canary types
    And each type has exactly 3 probes

  Scenario: Format canary verified correctly
    Given a canary system is initialized
    And a format canary that requires ending with "COMPLETE"
    When a response ending with "COMPLETE" is verified
    Then the canary check passes
    When a response not ending with "COMPLETE" is verified
    Then the canary check fails

  Scenario: Report computes per-type pass rates
    Given a canary system is initialized
    And an agent with a mock adapter
    When canary probes are executed against the agent
    Then the report contains per-type pass rates for each type present
    And the overall pass rate is between 0 and 1
