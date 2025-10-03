"""
Speculative Decoding Pipeline

Orchestrates the speculative decoding loop:
1. Use draft model to propose up to K tokens
2. Verify with base model; accept longest prefix
3. If accepted_len == 0, advance by 1 token using base model
4. Repeat until max_tokens or stop condition
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from .fake_lm import create_fake_lm
from .hf_wrappers import create_tiny_hf_wrapper
from .interfaces import LanguageModel, SpeculativeDecoder


class SpeculativePipeline(SpeculativeDecoder):
    """Main pipeline for speculative decoding with dependency injection."""

    def __init__(
        self,
        base_lm: Optional[LanguageModel] = None,
        draft_lm: Optional[LanguageModel] = None,
        config_path: Optional[str] = None,
        base_model: Optional[str] = None,
        draft_model: Optional[str] = None,
        max_draft: int = 4,
        device: str = "auto",
        seed: Optional[int] = None,
        implementation: Optional[str] = None,
        force_device: Optional[str] = None,
    ):
        """
        Initialize the speculative decoding pipeline.

        Args:
            base_lm: Base language model (if None, will be created)
            draft_lm: Draft language model (if None, will be created)
            config_path: Path to YAML configuration file
            base_model: Base model name (overrides config)
            draft_model: Draft model name (overrides config)
            max_draft: Maximum draft tokens per step (overrides config)
            device: Device to run on (overrides config)
            seed: Random seed for reproducibility
            implementation: Implementation type ("fake" or "hf")
            force_device: Force both models to same device ("cpu", "mps")
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Override config with provided parameters
        if base_model:
            self.config["base_model"] = base_model
        if draft_model:
            self.config["draft_model"] = draft_model
        if max_draft:
            self.config["max_draft"] = max_draft
        if device != "auto":
            self.config["device"] = device
        if implementation:
            self.config["implementation"] = implementation
        if force_device:
            self.config["force_device"] = force_device

        # Set random seed if provided
        if seed is not None:
            self._set_seed(seed)
            self.config["seed"] = seed

        self.max_draft = self.config["max_draft"]
        self.device = self._select_device(self.config["device"])
        self.implementation = self.config.get("implementation", "fake")
        self.force_device = self.config.get("force_device")

        # Log startup configuration
        self._log_startup_config()

        # Initialize models with dependency injection
        self.base_lm = base_lm or self._create_base_model()
        self.draft_lm = draft_lm or self._create_draft_model()

        # Check tokenizer compatibility
        self._check_compatibility()

        # Metrics tracking
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "base_model": "facebook/opt-125m",
            "draft_model": "distilgpt2",
            "max_draft": 4,
            "temperature": 0.7,
            "do_sample": True,
            "device": "auto",
            "seed": 1234,
            "max_new_tokens": 64,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "implementation": "fake",
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                default_config.update(config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def _log_startup_config(self) -> None:
        """Log startup configuration summary."""
        device = self.force_device or self.device
        dtype = "float16" if device == "mps" else "float32"

        self.logger.info(
            f"Startup config: impl={self.implementation}, device={device}, "
            f"dtype={dtype}, base_model={self.config['base_model']}, "
            f"draft_model={self.config['draft_model']}, max_draft={self.max_draft}, "
            f"max_tokens={self.config.get('max_new_tokens', 64)}"
        )

    def _create_base_model(self) -> LanguageModel:
        """Create the base language model based on implementation."""
        if self.implementation == "fake":
            return create_fake_lm(
                model_name=f"fake-base-{self.config['base_model']}",
                seed=self.config.get("seed"),
            )
        elif self.implementation == "hf":
            # Use tiny models for HF implementation
            device = self.force_device or self.device
            return create_tiny_hf_wrapper(
                model_name=self.config["base_model"],
                device=device,
                max_memory_mb=500,
            )
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _create_draft_model(self) -> LanguageModel:
        """Create the draft language model based on implementation."""
        if self.implementation == "fake":
            return create_fake_lm(
                model_name=f"fake-draft-{self.config['draft_model']}",
                seed=self.config.get("seed"),
                acceptance_rate=0.7,  # Simulate realistic acceptance rate
            )
        elif self.implementation == "hf":
            # Use tiny models for HF implementation with shared tokenizer
            device = self.force_device or self.device
            base_lm = self.base_lm
            shared_tokenizer = None
            if hasattr(base_lm, "_tokenizer"):
                shared_tokenizer = base_lm._tokenizer

            return create_tiny_hf_wrapper(
                model_name=self.config["draft_model"],
                device=device,
                max_memory_mb=500,
                tokenizer=shared_tokenizer,
            )
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        import random

        random.seed(seed)
        self.logger.info(f"Set random seed to {seed}")

    def _select_device(self, device: str) -> str:
        """Select the best available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _check_compatibility(self) -> None:
        """Check tokenizer compatibility between draft and base models."""
        base_info = self.base_lm.get_tokenizer_info()
        draft_info = self.draft_lm.get_tokenizer_info()

        # Check if tokenizers are compatible
        compatible = (
            base_info["vocab_size"] == draft_info["vocab_size"]
            and base_info["pad_token_id"] == draft_info["pad_token_id"]
            and base_info["eos_token_id"] == draft_info["eos_token_id"]
        )

        if not compatible:
            self.logger.warning(
                f"Tokenizer incompatibility detected: "
                f"base={base_info}, draft={draft_info}"
            )
            self.logger.warning(
                "Different tokenizer families detected. This may reduce "
                "acceptance rates."
            )

    def _find_accepted_length(
        self, proposed_tokens: torch.Tensor, base_tokens: torch.Tensor
    ) -> int:
        """
        Find the length of the longest matching prefix between proposed and base tokens.

        Args:
            proposed_tokens: Proposed token IDs [batch_size, num_proposed]
            base_tokens: Base model token IDs [batch_size, num_base]

        Returns:
            Length of accepted prefix
        """
        # For simplicity, we'll work with the first batch item
        proposed = proposed_tokens[0].cpu().numpy()
        base = base_tokens[0].cpu().numpy()

        # Find the minimum length to compare
        min_len = min(len(proposed), len(base))

        # Count consecutive matching tokens
        accepted_len = 0
        for i in range(min_len):
            if proposed[i] == base[i]:
                accepted_len += 1
            else:
                break

        return accepted_len

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            do_sample: Whether to use sampling (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated text and metrics
        """
        start_time = time.time()

        # Use provided parameters or fall back to config
        max_tokens = max_tokens or self.config["max_new_tokens"]
        temperature = temperature or self.config["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["do_sample"]

        # Reset metrics
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
        }

        try:
            # Tokenize input prompt
            input_ids = self.base_lm.encode(prompt)
            # Move to device if needed (but only if device is not "auto")
            if (
                self.device != "auto"
                and self.device != "cpu"
                and input_ids.device == torch.device("cpu")
            ):
                input_ids = input_ids.to(self.device)

            # Initialize generation state
            generated_tokens: list[int] = []
            current_input = input_ids.clone()
            step = 0

            self.logger.info(
                f"Starting speculative decoding: prompt=\"{prompt[:50]}...\", "
                f"max_tokens={max_tokens}"
            )

            while (
                len(generated_tokens) < max_tokens and step < max_tokens * 2
            ):  # Safety limit
                step += 1
                step_start = time.time()

                # Step 1: Generate draft tokens
                draft_tokens, _ = self.draft_lm.generate_tokens(
                    current_input,
                    max_new_tokens=self.max_draft,
                    temperature=temperature,
                    do_sample=do_sample,
                    **kwargs,
                )

                # Step 2: Verify with base model
                base_tokens, _ = self.base_lm.generate_tokens(
                    current_input,
                    max_new_tokens=self.max_draft,
                    temperature=temperature,
                    do_sample=do_sample,
                    **kwargs,
                )

                # Find accepted length
                accepted_len = self._find_accepted_length(draft_tokens, base_tokens)
                accepted_tokens = (
                    draft_tokens[:, :accepted_len]
                    if accepted_len > 0
                    else torch.empty(draft_tokens.shape[0], 0, dtype=draft_tokens.dtype)
                )

                # Update metrics
                self.metrics["total_proposed"] += draft_tokens.shape[1]
                self.metrics["total_accepted"] += accepted_len
                verification_time_ms = (time.time() - step_start) * 1000
                self.metrics["total_verification_time_ms"] += verification_time_ms

                # Step 3: Handle results
                if accepted_len > 0:
                    # Accept the proposed tokens (but respect max_tokens limit)
                    if accepted_tokens.numel() > 0:
                        # Calculate how many tokens we can still accept
                        remaining_tokens = max_tokens - len(generated_tokens)
                        if remaining_tokens > 0:
                            # Only accept up to the remaining limit
                            tokens_to_accept = min(
                                accepted_tokens.shape[1], remaining_tokens
                            )
                            accepted_tokens_limited = accepted_tokens[
                                :, :tokens_to_accept
                            ]
                            generated_tokens.extend(
                                accepted_tokens_limited[0].cpu().tolist()
                            )
                            current_input = torch.cat(
                                [current_input, accepted_tokens_limited], dim=1
                            )

                    self.logger.info(
                        f"Step {step}: accepted {accepted_len}/"
                        f"{draft_tokens.shape[1]} tokens, "
                        f"total: {len(generated_tokens)}/{max_tokens}"
                    )
                else:
                    # Fallback: generate one token with base model
                    # (if we haven't reached max_tokens)
                    remaining_tokens = max_tokens - len(generated_tokens)
                    if remaining_tokens > 0:
                        fallback_tokens, _ = self.base_lm.generate_tokens(
                            current_input,
                            max_new_tokens=1,
                            temperature=temperature,
                            do_sample=do_sample,
                            **kwargs,
                        )

                        if fallback_tokens.numel() > 0:
                            generated_tokens.extend(fallback_tokens[0].cpu().tolist())
                            current_input = torch.cat(
                                [current_input, fallback_tokens], dim=1
                            )

                    self.metrics["total_generation_time_ms"] += (
                        time.time() - step_start
                    ) * 1000

                    self.logger.info(
                        f"Step {step}: fallback generated 1 token, "
                        f"total: {len(generated_tokens)}/{max_tokens}"
                    )

                # Check for early stopping
                if len(generated_tokens) >= max_tokens:
                    self.logger.debug(
                        f"Early stopping: {len(generated_tokens)} >= {max_tokens}"
                    )
                    break

                # Check for EOS token
                base_info = self.base_lm.get_tokenizer_info()
                if (
                    generated_tokens
                    and generated_tokens[-1] == base_info["eos_token_id"]
                ):
                    self.logger.info("EOS token generated, stopping")
                    break

                step_time = (time.time() - step_start) * 1000
                self.logger.debug(f"Step {step} completed in {step_time:.2f}ms")

            # Finalize results
            self.metrics["total_steps"] = step
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["total_generation_time_ms"] += total_time_ms

            # Decode generated text
            if isinstance(generated_tokens, list):
                generated_text = self.base_lm.decode(torch.tensor([generated_tokens]))
            else:
                generated_text = self.base_lm.decode(generated_tokens)

            # Calculate final metrics
            acceptance_rate = (
                self.metrics["total_accepted"] / self.metrics["total_proposed"]
                if self.metrics["total_proposed"] > 0
                else 0.0
            )
            tokens_per_sec = (
                len(generated_tokens) / (total_time_ms / 1000)
                if total_time_ms > 0
                else 0.0
            )

            result = {
                "text": generated_text,
                "generated_tokens": generated_tokens,
                "latency_ms": total_time_ms,
                "proposed": self.metrics["total_proposed"],
                "accepted": self.metrics["total_accepted"],
                "acceptance_rate": acceptance_rate,
                "tokens_per_sec": tokens_per_sec,
                "steps": step,
                "verification_time_ms": self.metrics["total_verification_time_ms"],
                "generation_time_ms": self.metrics["total_generation_time_ms"],
                "impl": self.implementation,
                "device": self.force_device or self.device,
                "base_model": self.config["base_model"],
                "draft_model": self.config["draft_model"],
                "dtype": (
                    "float16"
                    if (self.force_device or self.device) == "mps"
                    else "float32"
                ),
            }

            self.logger.info(
                f"Speculative decoding completed: {len(generated_tokens)} tokens "
                f"in {total_time_ms:.2f}ms "
                f"({tokens_per_sec:.2f} tokens/sec, "
                f"acceptance_rate={acceptance_rate:.3f})"
            )

            # Cleanup MPS cache if available
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"Speculative decoding failed: {e}")
            raise
