"""
Unit and Integration Tests for Deterministic Mode

Tests that verify deterministic mode behavior:
1. Perfect draft (draft==target) produces identical output to vanilla decoding
2. Heavy repetition tokens are not filtered out in deterministic mode
3. Duplication detection is disabled in deterministic mode
4. Max tokens cap prevents over-generation
"""

import logging
import os
import sys
from pathlib import Path

import torch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from transformers import AutoTokenizer

from specdec.core.batch_handlers import AcceptanceHandler
from specdec.core.pipeline import SpeculativePipeline
from specdec.models.hf_wrappers import HFWrapper
from specdec.policies.controllers import create_controller
from specdec.policies.policies import create_policy
from specdec.utils.deterministic import set_deterministic_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torch to use 1 thread to avoid potential bus errors on macOS
torch.set_num_threads(1)


class TestDeterministicMode:
    """Test suite for deterministic mode correctness."""

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure deterministic mode is set
        set_deterministic_mode(seed=42, device="cpu")
        os.environ["SPECDEC_DETERMINISTIC"] = "1"
        os.environ["SPECDEC_ENABLE_KV_APPEND"] = "1"

    def test_duplication_detection_disabled_in_deterministic_mode(self):
        """Test that duplication detection is disabled when deterministic_mode=True."""
        # Create a mock policy
        policy = create_policy("longest_prefix", verify_backend="torch")

        # Create handler with deterministic_mode=True
        handler = AcceptanceHandler(
            policy=policy,
            tokenizer=None,
            base_lm=None,
            deterministic_mode=True,
        )

        # Test with heavy repetition (should NOT be filtered)
        accepted_tokens = [5087, 5087, 5087, 5087]
        generated_so_far = [5087, 5087, 5087, 5087]

        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=1,
        )

        # In deterministic mode, all tokens should be returned unchanged
        assert result == accepted_tokens, (
            f"Duplication detection should be disabled in deterministic mode. "
            f"Expected {accepted_tokens}, got {result}"
        )

    def test_duplication_detection_enabled_in_non_deterministic_mode(self):
        """Test that duplication detection is enabled when deterministic_mode=False."""
        # Create a mock policy
        policy = create_policy("longest_prefix", verify_backend="torch")

        # Create handler with deterministic_mode=False
        handler = AcceptanceHandler(
            policy=policy,
            tokenizer=None,
            base_lm=None,
            deterministic_mode=False,
        )

        # Test with phrase repetition (should be filtered)
        accepted_tokens = [5087, 5087, 5087, 5087]
        generated_so_far = [5087, 5087, 5087, 5087]

        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=1,
        )

        # In non-deterministic mode, phrase repetition should be filtered
        assert result == [], (
            f"Duplication detection should filter phrase repetition in non-deterministic mode. "
            f"Expected [], got {result}"
        )

    def test_perfect_draft_deterministic_matches_vanilla(self):
        """Test that perfect draft (draft==target) produces identical output to vanilla."""
        model_name = "sshleifer/tiny-gpt2"

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = HFWrapper(model_name=model_name, device="cpu", tokenizer=tokenizer)
        draft_model = HFWrapper(
            model_name=model_name, device="cpu", tokenizer=tokenizer
        )

        # Vanilla generation
        prompt = "The quick brown fox"
        input_ids = base_model.encode(prompt)
        vanilla_tokens = []
        current_input = input_ids.clone()

        for _ in range(10):
            tokens, _ = base_model.generate_tokens(
                current_input, max_new_tokens=1, temperature=1.0, do_sample=False
            )
            if tokens.numel() == 0:
                break
            token_id = tokens[0, 0].item()
            vanilla_tokens.append(token_id)
            current_input = torch.cat([current_input, tokens[:, :1]], dim=1)

        # Speculative decoding with deterministic mode
        pipeline = SpeculativePipeline(
            base_lm=base_model,
            draft_lm=draft_model,
            max_draft=4,
            device="cpu",
            seed=42,
            policy="longest_prefix",
            controller="fixed",
            controller_params={"k": 4},
            max_seq_len=2048,
        )

        # Ensure deterministic mode is enabled
        assert (
            pipeline.deterministic_mode
        ), "Pipeline should have deterministic_mode=True"

        results = pipeline.generate_batch(
            prompts=[prompt],
            max_tokens=10,
            temperature=1.0,
            do_sample=False,
        )

        specdec_tokens = results[0].get("generated_tokens", [])

        # Should match exactly
        assert vanilla_tokens == specdec_tokens, (
            f"SpecDec output should match vanilla in deterministic mode. "
            f"Vanilla: {vanilla_tokens}, SpecDec: {specdec_tokens}"
        )

    def test_max_tokens_cap_prevents_over_generation(self):
        """Test that max_tokens cap prevents generating more tokens than requested."""
        model_name = "sshleifer/tiny-gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = HFWrapper(model_name=model_name, device="cpu", tokenizer=tokenizer)
        draft_model = HFWrapper(
            model_name=model_name, device="cpu", tokenizer=tokenizer
        )

        pipeline = SpeculativePipeline(
            base_lm=base_model,
            draft_lm=draft_model,
            max_draft=4,  # k=4
            device="cpu",
            seed=42,
            policy="longest_prefix",
            controller="fixed",
            controller_params={"k": 4},
            max_seq_len=2048,
        )

        # Request only 5 tokens, but k=4 means we could generate 4+4=8 without cap
        max_tokens = 5
        results = pipeline.generate_batch(
            prompts=["Test prompt"],
            max_tokens=max_tokens,
            temperature=1.0,
            do_sample=False,
        )

        specdec_tokens = results[0].get("generated_tokens", [])

        # Should not exceed max_tokens
        assert len(specdec_tokens) <= max_tokens, (
            f"SpecDec should not generate more than {max_tokens} tokens. "
            f"Generated {len(specdec_tokens)} tokens: {specdec_tokens}"
        )

        # Should generate exactly max_tokens (unless EOS encountered)
        assert len(specdec_tokens) == max_tokens, (
            f"SpecDec should generate exactly {max_tokens} tokens in deterministic mode. "
            f"Generated {len(specdec_tokens)} tokens: {specdec_tokens}"
        )

    def test_heavy_repetition_not_filtered_in_deterministic_mode(self):
        """Test that heavy repetition (like token 5087) is not filtered in deterministic mode."""
        model_name = "sshleifer/tiny-gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = HFWrapper(model_name=model_name, device="cpu", tokenizer=tokenizer)
        draft_model = HFWrapper(
            model_name=model_name, device="cpu", tokenizer=tokenizer
        )

        # Use a prompt that produces repetitive tokens
        prompt = "The quick brown fox"

        # Vanilla generation
        input_ids = base_model.encode(prompt)
        vanilla_tokens = []
        current_input = input_ids.clone()

        for _ in range(20):
            tokens, _ = base_model.generate_tokens(
                current_input, max_new_tokens=1, temperature=1.0, do_sample=False
            )
            if tokens.numel() == 0:
                break
            token_id = tokens[0, 0].item()
            vanilla_tokens.append(token_id)
            current_input = torch.cat([current_input, tokens[:, :1]], dim=1)

        # Speculative decoding with deterministic mode
        pipeline = SpeculativePipeline(
            base_lm=base_model,
            draft_lm=draft_model,
            max_draft=4,
            device="cpu",
            seed=42,
            policy="longest_prefix",
            controller="fixed",
            controller_params={"k": 4},
            max_seq_len=2048,
        )

        results = pipeline.generate_batch(
            prompts=[prompt],
            max_tokens=20,
            temperature=1.0,
            do_sample=False,
        )

        specdec_tokens = results[0].get("generated_tokens", [])

        # Should match exactly even with repetition
        assert vanilla_tokens == specdec_tokens, (
            f"SpecDec should match vanilla even with heavy repetition in deterministic mode. "
            f"Vanilla: {vanilla_tokens}, SpecDec: {specdec_tokens}"
        )


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v"])
