Feature: Agent Authentication
  As a security platform
  I need to verify that an agent is who it claims to be
  So that I can prevent unauthorized agent substitution

  Scenario: Authentic agent passes verification
    Given an agent "alpha" with an established baseline
    And a new signature from the same agent
    When authentication is performed
    Then the agent is verified as authentic
    And the confidence score is above 0.5

  Scenario: Impostor agent fails verification
    Given an agent "alpha" with an established baseline
    And a signature from a different agent "beta"
    When authentication is performed against alpha's baseline
    Then the agent is not verified as authentic
