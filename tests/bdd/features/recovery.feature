Feature: Recovery
  As a security platform
  I need to recover compromised agents to known-good state
  So that normal operations can resume

  Scenario: Successful recovery after compromise
    Given an agent "alpha" that was flagged as compromised
    And a set of clean recovery prompts
    When recovery is attempted
    Then the recovery succeeds
    And the new baseline is close to the old baseline
