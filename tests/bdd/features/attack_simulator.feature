Feature: Adversarial Attack Simulator
  As a security platform
  I need to simulate adversarial attacks against the detection system
  So that I can measure detection resilience across 4 attack strategies

  Scenario: All 4 attack types produce results
    Given an attack simulator with all detection engines
    And a target agent with a baseline signature
    When all 4 attacks are simulated
    Then each attack produces an AttackResult
    And each result has a valid attack type

  Scenario: Detection rates are computed correctly
    Given an attack simulator with all detection engines
    And a target agent with a baseline signature
    When all 4 attacks are simulated
    Then each result has detection_rate in [0, 1]
    And each result has evasion_rate = 1 - detection_rate

  Scenario: Summary report aggregates all attacks
    Given an attack simulator with all detection engines
    And a target agent with a baseline signature
    When all 4 attacks are simulated
    And a summary report is generated
    Then the report contains all 4 attack types
    And the report has overall detection and evasion rates
