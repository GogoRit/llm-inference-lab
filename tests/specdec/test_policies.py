"""
Unit tests for acceptance policies.
"""

import pytest
import torch

from specdec.policies.policies import (
    ConfidenceThresholdPolicy,
    LongestPrefixPolicy,
    TopKAgreementPolicy,
    TypicalAcceptancePolicy,
    create_policy,
)


class TestLongestPrefixPolicy:
    """Test longest prefix acceptance policy."""

    def test_exact_match(self):
        """Test when all proposed tokens match base tokens."""
        policy = LongestPrefixPolicy()
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2, 3, 4]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 4
        assert info["policy"] == "longest_prefix"
        assert info["accepted_len"] == 4
        assert info["proposed_len"] == 4
        assert info["base_len"] == 4

    def test_partial_match(self):
        """Test when only some proposed tokens match."""
        policy = LongestPrefixPolicy()
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2, 5, 6]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 2
        assert info["accepted_len"] == 2

    def test_no_match(self):
        """Test when no proposed tokens match."""
        policy = LongestPrefixPolicy()
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[5, 6, 7, 8]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 0
        assert info["accepted_len"] == 0

    def test_different_lengths(self):
        """Test when proposed and base have different lengths."""
        policy = LongestPrefixPolicy()
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3, 4, 5]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 3
        assert info["accepted_len"] == 3


class TestConfidenceThresholdPolicy:
    """Test confidence threshold acceptance policy."""

    def test_high_confidence(self):
        """Test with high confidence scores."""
        policy = ConfidenceThresholdPolicy(tau=0.5)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        # Create high confidence logits
        logits = torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]])

        accepted_len, info = policy.accept_tokens(proposed, base, logits)

        assert accepted_len == 3
        assert info["policy"] == "conf_threshold"
        assert info["tau"] == 0.5

    def test_low_confidence(self):
        """Test with low confidence scores."""
        policy = ConfidenceThresholdPolicy(tau=0.8)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        # Create low confidence logits
        logits = torch.tensor([[[0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]])

        accepted_len, info = policy.accept_tokens(proposed, base, logits)

        assert accepted_len == 0
        assert info["accepted_len"] == 0

    def test_no_logits_fallback(self):
        """Test fallback to longest prefix when no logits provided."""
        policy = ConfidenceThresholdPolicy(tau=0.5)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 3
        assert info["policy"] == "longest_prefix"  # Fallback policy


class TestTopKAgreementPolicy:
    """Test top-k agreement acceptance policy."""

    def test_agreement_in_topk(self):
        """Test when proposed token is in base model's top-k."""
        policy = TopKAgreementPolicy(k=3)
        proposed = torch.tensor([[0, 1, 2]])  # Tokens 0, 1, 2
        base = torch.tensor([[0, 1, 2]])

        # Create logits where proposed tokens (0, 1, 2) are in top-k
        proposed_logits = torch.tensor(
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        )
        base_logits = torch.tensor(
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        )

        accepted_len, info = policy.accept_tokens(
            proposed, base, proposed_logits, base_logits
        )

        assert accepted_len == 3
        assert info["policy"] == "topk_agree"
        assert info["k"] == 3

    def test_no_agreement_in_topk(self):
        """Test when proposed token is not in base model's top-k."""
        policy = TopKAgreementPolicy(k=2)
        proposed = torch.tensor([[0, 1, 2]])  # Tokens 0, 1, 2
        base = torch.tensor([[0, 1, 2]])

        # Create logits where proposed tokens (0, 1, 2) are not in base model's top-2
        # Base model top-2 should be [2, 1] for all positions,
        # but proposed tokens are [0, 1, 2]
        proposed_logits = torch.tensor(
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        )
        base_logits = torch.tensor(
            [[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]]
        )

        accepted_len, info = policy.accept_tokens(
            proposed, base, proposed_logits, base_logits
        )

        # Token 0 is in base model's top-2 [2, 0], so it accepts 1 token
        assert accepted_len == 1
        assert info["accepted_len"] == 1

    def test_no_logits_fallback(self):
        """Test fallback to longest prefix when no logits provided."""
        policy = TopKAgreementPolicy(k=3)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 3
        assert info["policy"] == "longest_prefix"  # Fallback policy


class TestTypicalAcceptancePolicy:
    """Test typical acceptance policy."""

    def test_high_probability(self):
        """Test with high typical acceptance probability."""
        policy = TypicalAcceptancePolicy(p=0.5)
        proposed = torch.tensor([[0, 1, 2]])  # Tokens 0, 1, 2
        base = torch.tensor([[0, 1, 2]])

        # Create logits with high probability for proposed tokens (0, 1, 2)
        proposed_logits = torch.tensor(
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        )
        base_logits = torch.tensor(
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        )

        accepted_len, info = policy.accept_tokens(
            proposed, base, proposed_logits, base_logits
        )

        assert accepted_len == 3
        assert info["policy"] == "typical"
        assert info["p"] == 0.5

    def test_low_probability(self):
        """Test with low typical acceptance probability."""
        policy = TypicalAcceptancePolicy(p=0.9)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        # Create logits with low probability for proposed tokens
        proposed_logits = torch.tensor(
            [[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]]
        )
        base_logits = torch.tensor(
            [[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]]
        )

        accepted_len, info = policy.accept_tokens(
            proposed, base, proposed_logits, base_logits
        )

        assert accepted_len == 0
        assert info["accepted_len"] == 0

    def test_no_logits_fallback(self):
        """Test fallback to longest prefix when no logits provided."""
        policy = TypicalAcceptancePolicy(p=0.5)
        proposed = torch.tensor([[1, 2, 3]])
        base = torch.tensor([[1, 2, 3]])

        accepted_len, info = policy.accept_tokens(proposed, base)

        assert accepted_len == 3
        assert info["policy"] == "longest_prefix"  # Fallback policy


class TestCreatePolicy:
    """Test policy creation function."""

    def test_create_longest_prefix(self):
        """Test creating longest prefix policy."""
        policy = create_policy("longest_prefix")
        assert isinstance(policy, LongestPrefixPolicy)

    def test_create_conf_threshold(self):
        """Test creating confidence threshold policy."""
        policy = create_policy("conf_threshold", tau=0.7)
        assert isinstance(policy, ConfidenceThresholdPolicy)
        assert policy.tau == 0.7

    def test_create_topk_agree(self):
        """Test creating top-k agreement policy."""
        policy = create_policy("topk_agree", k=10)
        assert isinstance(policy, TopKAgreementPolicy)
        assert policy.k == 10

    def test_create_typical(self):
        """Test creating typical acceptance policy."""
        policy = create_policy("typical", p=0.8)
        assert isinstance(policy, TypicalAcceptancePolicy)
        assert policy.p == 0.8

    def test_invalid_policy(self):
        """Test creating invalid policy raises error."""
        with pytest.raises(ValueError, match="Unknown policy"):
            create_policy("invalid_policy")
