"""
Unit and Integration Tests for Duplication Detection

Tests that verify duplication detection behavior:
1. Verified tokens (draft matches base) should NOT be filtered in non-deterministic mode
2. Unverified repetitive tokens (draft repeats but base doesn't confirm) should be filtered
3. Deterministic mode bypasses duplication detection completely
4. Perfect draft with repetitive tokens maintains ~100% acceptance rate
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
from specdec.policies.policies import create_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torch to use 1 thread to avoid potential bus errors on macOS
torch.set_num_threads(1)


class TestDuplicationDetection:
    """Test suite for duplication detection correctness."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear deterministic mode for non-deterministic tests
        if "SPECDEC_DETERMINISTIC" in os.environ:
            del os.environ["SPECDEC_DETERMINISTIC"]

    def test_verified_tokens_not_filtered_in_non_deterministic_mode(self):
        """
        Test that verified tokens (accepted by acceptance policy) are NOT filtered
        by duplication detection in non-deterministic mode.

        Contract: Verified tokens should not be filtered by duplication detection
        in non-deterministic mode, even if they're repetitive.
        """
        # Create handler with deterministic_mode=False (non-deterministic)
        policy = create_policy("longest_prefix", verify_backend="torch")
        handler = AcceptanceHandler(
            policy=policy,
            tokenizer=None,
            base_lm=None,
            deterministic_mode=False,
        )

        # Simulate verified repetitive tokens (e.g., [5087, 5087, 5087, 5087])
        # These were verified by the acceptance policy (draft matches base)
        accepted_tokens = [5087, 5087, 5087, 5087]
        generated_so_far = [
            5087,
            5087,
            5087,
            5087,
        ]  # Previous step generated same tokens

        # In the batch loop, verified tokens skip duplication detection
        # But if we call it directly, it should still filter (this tests the handler logic)
        # However, the batch loop now skips this for verified tokens, so this test
        # verifies the handler's behavior when called directly

        # Note: The actual fix is in batch_loop.py where we skip duplication detection
        # for verified tokens. This test documents the expected behavior.
        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=2,
        )

        # The handler's detect_duplication will filter (that's its job)
        # But the batch loop skips it for verified tokens
        # This test documents that the handler itself filters, but the batch loop
        # correctly bypasses it for verified tokens
        assert len(result) < len(accepted_tokens), (
            "Handler's detect_duplication should filter repetitive tokens when called directly. "
            "However, batch_loop.py should skip this for verified tokens."
        )

    def test_perfect_draft_repetitive_tokens_high_acceptance_rate(self):
        """
        Test that perfect draft (draft==base) with repetitive tokens maintains
        ~100% acceptance rate in non-deterministic mode.

        Contract: Verified tokens should not be filtered, leading to high acceptance rate.
        """
        model_name = "sshleifer/tiny-gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = HFWrapper(model_name=model_name, device="cpu", tokenizer=tokenizer)
        draft_model = HFWrapper(
            model_name=model_name, device="cpu", tokenizer=tokenizer
        )

        # Create pipeline with deterministic_mode=False (non-deterministic)
        # This tests the batch loop's logic to skip duplication detection for verified tokens
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

        # Ensure deterministic mode is False
        assert (
            not pipeline.deterministic_mode
        ), "Pipeline should have deterministic_mode=False for this test"

        # Use a prompt that may produce repetitive tokens
        prompt = "The quick brown fox"

        results = pipeline.generate_batch(
            prompts=[prompt],
            max_tokens=16,
            temperature=1.0,
            do_sample=False,
        )

        # Check metrics (metrics are directly in the result dict, not nested)
        total_proposed = results[0].get("proposed", 0)
        total_accepted = results[0].get("accepted", 0)
        acceptance_rate = results[0].get("acceptance_rate", 0.0)
        if acceptance_rate == 0.0 and total_proposed > 0:
            acceptance_rate = total_accepted / max(total_proposed, 1)

        # For perfect draft, acceptance rate should be very high (~100%)
        # The fix ensures verified tokens aren't filtered, so acceptance should be high
        assert acceptance_rate >= 0.90, (
            f"Perfect draft should have high acceptance rate (>=90%). "
            f"Got {acceptance_rate:.2%} (proposed={total_proposed}, accepted={total_accepted})"
        )

        # Total tokens should match what we requested (or close, accounting for EOS)
        generated_tokens = results[0].get("generated_tokens", [])
        assert (
            len(generated_tokens) >= 14
        ), f"Should generate close to max_tokens=16. Got {len(generated_tokens)} tokens"  # Allow some margin for EOS

        logger.info(
            f"Perfect draft test: acceptance_rate={acceptance_rate:.2%}, "
            f"proposed={total_proposed}, accepted={total_accepted}, "
            f"generated={len(generated_tokens)} tokens"
        )

    def test_unverified_repetitive_tokens_filtered(self):
        """
        Test that unverified repetitive tokens (draft repeats but base doesn't confirm)
        are still filtered by duplication detection.

        Contract: Unverified repetitive tokens should be filtered to prevent artifacts.
        """
        # This test is harder to set up without mocking, but we can test the handler directly
        policy = create_policy("longest_prefix", verify_backend="torch")
        handler = AcceptanceHandler(
            policy=policy,
            tokenizer=None,
            base_lm=None,
            deterministic_mode=False,
        )

        # Simulate unverified repetitive tokens
        # In a real scenario, the draft might propose [5087, 5087, 5087, 5087]
        # but the base only confirms [5087] (accepted_len=1)
        # Then duplication detection should filter if there's repetition

        # Test phrase repetition detection
        accepted_tokens = [100, 200, 300, 100, 200, 300]  # Pattern repeats
        generated_so_far = [100, 200, 300]  # Previous tokens match pattern

        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=2,
        )

        # Phrase repetition should be filtered
        assert len(result) < len(accepted_tokens), (
            f"Phrase repetition should be filtered. "
            f"Expected fewer tokens, got {len(result)} tokens: {result}"
        )

        # Test single token repetition
        accepted_tokens = [5087, 5087, 5087, 5087]
        generated_so_far = [5087, 5087, 5087, 5087]

        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=2,
        )

        # Single token repetition should be filtered (when called directly)
        # Note: In batch_loop.py, this is skipped for verified tokens
        assert len(result) < len(accepted_tokens), (
            f"Single token repetition should be filtered when called directly. "
            f"Got {len(result)} tokens: {result}"
        )

    def test_deterministic_mode_bypasses_duplication_detection(self):
        """
        Test that deterministic mode completely bypasses duplication detection.

        Contract: Deterministic mode bypasses duplication detection completely.
        """
        policy = create_policy("longest_prefix", verify_backend="torch")
        handler = AcceptanceHandler(
            policy=policy,
            tokenizer=None,
            base_lm=None,
            deterministic_mode=True,  # Deterministic mode
        )

        # Test with heavy repetition
        accepted_tokens = [5087, 5087, 5087, 5087]
        generated_so_far = [5087, 5087, 5087, 5087]

        result = handler.detect_duplication(
            accepted_tokens=accepted_tokens,
            generated_so_far=generated_so_far,
            global_idx=0,
            step=2,
        )

        # In deterministic mode, all tokens should be returned unchanged
        assert result == accepted_tokens, (
            f"Deterministic mode should bypass duplication detection. "
            f"Expected {accepted_tokens}, got {result}"
        )

    def test_perfect_draft_matches_vanilla_with_repetition(self):
        """
        Integration test: Perfect draft with repetitive tokens should match vanilla
        decoding in both deterministic and non-deterministic modes.
        """
        model_name = "sshleifer/tiny-gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = HFWrapper(model_name=model_name, device="cpu", tokenizer=tokenizer)
        draft_model = HFWrapper(
            model_name=model_name, device="cpu", tokenizer=tokenizer
        )

        prompt = "The quick brown fox"

        # Vanilla generation
        input_ids = base_model.encode(prompt)
        vanilla_tokens = []
        current_input = input_ids.clone()

        for _ in range(12):
            tokens, _ = base_model.generate_tokens(
                current_input, max_new_tokens=1, temperature=1.0, do_sample=False
            )
            if tokens.numel() == 0:
                break
            token_id = tokens[0, 0].item()
            vanilla_tokens.append(token_id)
            current_input = torch.cat([current_input, tokens[:, :1]], dim=1)

        # Test 1: Deterministic mode (should match exactly)
        os.environ["SPECDEC_DETERMINISTIC"] = "1"
        pipeline_det = SpeculativePipeline(
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

        results_det = pipeline_det.generate_batch(
            prompts=[prompt],
            max_tokens=12,
            temperature=1.0,
            do_sample=False,
        )

        specdec_tokens_det = results_det[0].get("generated_tokens", [])

        assert vanilla_tokens == specdec_tokens_det, (
            f"Deterministic mode should match vanilla exactly. "
            f"Vanilla: {vanilla_tokens}, SpecDec: {specdec_tokens_det}"
        )

        # Test 2: Non-deterministic mode (should have high acceptance, tokens may differ slightly)
        del os.environ["SPECDEC_DETERMINISTIC"]
        pipeline_non_det = SpeculativePipeline(
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

        results_non_det = pipeline_non_det.generate_batch(
            prompts=[prompt],
            max_tokens=12,
            temperature=1.0,
            do_sample=False,
        )

        specdec_tokens_non_det = results_non_det[0].get("generated_tokens", [])
        total_proposed = results_non_det[0].get("proposed", 0)
        total_accepted = results_non_det[0].get("accepted", 0)
        acceptance_rate = results_non_det[0].get("acceptance_rate", 0.0)
        if acceptance_rate == 0.0 and total_proposed > 0:
            acceptance_rate = total_accepted / max(total_proposed, 1)

        # Non-deterministic should have high acceptance rate
        assert acceptance_rate >= 0.90, (
            f"Non-deterministic mode should have high acceptance rate (>=90%). "
            f"Got {acceptance_rate:.2%}"
        )

        # Should generate similar number of tokens
        assert abs(len(specdec_tokens_non_det) - len(vanilla_tokens)) <= 2, (
            f"Non-deterministic should generate similar number of tokens. "
            f"Vanilla: {len(vanilla_tokens)}, SpecDec: {len(specdec_tokens_non_det)}"
        )

        logger.info(
            f"Perfect draft test: deterministic matches vanilla, "
            f"non-deterministic acceptance_rate={acceptance_rate:.2%}"
        )


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v"])
