Feature: Dynamic Probe Generation
  As a security platform
  I need probes that are unpredictable and resist static prompt training
  So that compromised agents cannot anticipate and game the evaluation

  Scenario: Probe set is unpredictable across seeds
    Given a probe generator with seed 111
    And another probe generator with seed 222
    When both generate probe sets of size 15
    Then the overlap between probe texts is less than 50 percent

  Scenario: Category coverage is balanced
    Given a probe generator with seed 42
    When a probe set of size 15 is generated
    Then each category has at least 2 probes

  Scenario: Previously used probes are excluded
    Given a probe generator with seed 42
    And a first probe set of size 10
    When a second probe set of size 10 is generated excluding the first set's hashes
    Then no probe hashes from the first set appear in the second set
