Feature: Semantic Analysis for Gaming Detection
  As a security platform
  I need to detect when an agent games metrics by producing structurally similar
  but semantically different responses
  So that I can distinguish genuine consistency from metric gaming

  Scenario: Gaming detection catches structurally similar but semantically different
    Given a baseline response "Photosynthesis converts sunlight into chemical energy using chlorophyll"
    And a current response "Chocolate cake requires flour eggs sugar and butter for baking"
    And a structural similarity of 0.92
    When semantic analysis is performed
    Then the semantic gap is positive
    And gaming is detected

  Scenario: Genuine similarity passes
    Given a baseline response "The capital of France is Paris"
    And a current response "The capital of France is Paris"
    And a structural similarity of 0.95
    When semantic analysis is performed
    Then the semantic gap is near zero or negative
    And gaming is not detected

  Scenario: Semantic gap is positive when gaming occurs
    Given a baseline response "Machine learning uses algorithms to find patterns in data"
    And a current response "Baking bread requires kneading dough and letting it rise overnight"
    And a structural similarity of 0.88
    When semantic analysis is performed
    Then the semantic gap is greater than 0.3
