Feature: Temporal Drift Tracking
  As a security platform
  I need to detect temporal drift patterns over a sequence of agent snapshots
  So that I can identify gradual compromise, sudden jumps, and oscillations

  Scenario: Gradual drift detected over 8+ snapshots
    Given a baseline signature at the origin
    And 8 snapshots with gradually increasing distances
    When temporal drift is tracked
    Then the pattern is "gradual_accumulation"
    And drift velocity is positive

  Scenario: Sudden jump detected as anomaly
    Given a baseline signature at the origin
    And 8 snapshots with a sudden jump at index 3
    When temporal drift is tracked
    Then the pattern is "sudden_jump"
    And index 3 is flagged as an anomaly

  Scenario: Stable pattern classified correctly
    Given a baseline signature at the origin
    And 8 snapshots with constant distances
    When temporal drift is tracked
    Then the pattern is "stable"
    And no anomalies are detected
