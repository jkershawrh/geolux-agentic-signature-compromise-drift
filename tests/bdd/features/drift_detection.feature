Feature: Drift Detection
  As a security platform
  I need to detect when an agent's behavior drifts from its baseline
  So that I can identify potential compromise

  Scenario: No drift detected for healthy agent
    Given an agent "alpha" with a healthy baseline
    And a new set of healthy runs
    When drift is measured
    Then the drift magnitude is below 0.5
    And the drift is not significant

  Scenario: Drift detected after perturbation
    Given an agent "alpha" with a healthy baseline
    And a perturbed set of runs with different behavior
    When drift is measured
    Then the geodesic distance is above 0
    And the drift category is classified
