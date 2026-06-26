Feature: Measurement Security
  As a security platform
  I need to protect geometric signatures from reverse-engineering
  So that attackers cannot reconstruct or tamper with behavioral fingerprints

  Scenario: Signature encryption protects vector at rest
    Given an agent "gamma" with a geometric signature
    When the signature is encrypted
    Then the encrypted envelope differs from the raw vector
    And decrypting the envelope recovers the original vector

  Scenario: Commitment hash detects tampering
    Given an agent "gamma" with a geometric signature
    When the signature is encrypted
    And the commitment hash is verified against the original vector
    Then the verification succeeds
    But verification against a tampered vector fails

  Scenario: Drift obfuscation hides exact dimensions
    Given a drift measurement with known per-dimension values
    When the drift is obfuscated with noise
    Then the obfuscated dimensions differ from the originals
    And the severity classification is preserved
