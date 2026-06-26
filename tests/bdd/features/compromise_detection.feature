Feature: Compromise Detection
  As a security platform
  I need to detect when an agent has been compromised
  So that I can take corrective action

  Scenario: Compromised agent triggers alert
    Given an agent "alpha" with a healthy baseline
    And a drift measurement with high compromise probability
    When the compromise detector evaluates the drift
    Then an alert is generated
    And the alert severity is "warning" or "critical"

  Scenario: Healthy agent does not trigger alert
    Given an agent "alpha" with a healthy baseline
    And a drift measurement with low compromise probability
    When the compromise detector evaluates the drift
    Then no alert is generated
