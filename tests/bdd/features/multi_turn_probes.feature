Feature: Multi-Turn Behavioral Probes
  As a security platform
  I need to test conversation consistency across turns
  So that I can detect agents that lose context or deviate from instructions

  Scenario: Memory probe checks recall across turns
    Given a multi-turn prober with a mock conversational adapter
    And an agent for multi-turn testing
    When a memory probe conversation is executed
    Then the memory consistency score is above zero
    And the result contains 8 turns

  Scenario: Instruction persistence holds across turns
    Given a multi-turn prober with a mock conversational adapter
    And an agent for multi-turn testing
    When an instruction persistence probe conversation is executed
    Then the instruction persistence score is above zero
    And assistant responses after the first end with "DONE"

  Scenario: Coherence probe detects consistency
    Given a multi-turn prober with a mock conversational adapter
    And an agent for multi-turn testing
    When a coherence probe conversation is executed
    Then the behavioral coherence score is above zero
    And all scores are between 0 and 1
