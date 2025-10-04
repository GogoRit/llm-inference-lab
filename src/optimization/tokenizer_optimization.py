"""
Tokenizer Optimization Utilities

Provides optimized tokenization with caching, batching, and memory efficiency
for improved performance in local CPU/MPS development.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class OptimizedTokenizer:
    """Optimized tokenizer wrapper with caching and batching support."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        cache_size: int = 1000,
        batch_size: int = 32,
        enable_caching: bool = True,
        device: str = "auto",
    ):
        """
        Initialize optimized tokenizer.

        Args:
            tokenizer: Hugging Face tokenizer
            cache_size: Maximum number of cached tokenizations
            batch_size: Default batch size for batched operations
            enable_caching: Whether to enable tokenization caching
            device: Device to run on
        """
        self.tokenizer = tokenizer
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.device = self._select_device(device)

        # Initialize cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"OptimizedTokenizer initialized: cache_size={cache_size}, "
            f"batch_size={batch_size}, caching={enable_caching}, device={self.device}"
        )

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

    def _get_cache_key(self, text: str, **kwargs) -> str:
        """Generate cache key for tokenization parameters."""
        # Create a hashable key from text and kwargs
        key_parts = [text]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def _cache_result(self, key: str, result: Dict[str, Any]) -> None:
        """Cache tokenization result."""
        if not self.enable_caching:
            return

        # Remove oldest entries if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result

    def _get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached tokenization result."""
        if not self.enable_caching:
            return None

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[int], List[List[int]]]:
        """
        Optimized encoding with caching and batching support.

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            return_tensors: Format to return tensors in
            **kwargs: Additional tokenization parameters

        Returns:
            Encoded tokens
        """
        # Handle single text
        if isinstance(text, str):
            return self._encode_single(
                text, add_special_tokens, return_tensors, **kwargs
            )

        # Handle batch of texts
        return self._encode_batch(text, add_special_tokens, return_tensors, **kwargs)

    def _encode_single(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[int]]:
        """Encode single text with caching."""
        # Check cache
        cache_key = self._get_cache_key(
            text, add_special_tokens=add_special_tokens, **kwargs
        )
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            return self._format_result(cached_result, return_tensors)

        # Perform tokenization
        result = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs,
        )

        # Cache result
        self._cache_result(
            cache_key,
            {
                "tokens": result,
                "add_special_tokens": add_special_tokens,
                "return_tensors": return_tensors,
            },
        )

        return result

    def _encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[List[int]]]:
        """Encode batch of texts with optimization."""
        # Check if we can use cached results for some texts
        if self.enable_caching:
            cached_results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(
                    text, add_special_tokens=add_special_tokens, **kwargs
                )
                cached_result = self._get_cached_result(cache_key)

                if cached_result is not None:
                    cached_results.append((i, cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Process uncached texts
            if uncached_texts:
                uncached_results = self.tokenizer.encode(
                    uncached_texts,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    **kwargs,
                )

                # Cache uncached results
                for text, result in zip(uncached_texts, uncached_results):
                    cache_key = self._get_cache_key(
                        text, add_special_tokens=add_special_tokens, **kwargs
                    )
                    self._cache_result(
                        cache_key,
                        {
                            "tokens": result,
                            "add_special_tokens": add_special_tokens,
                            "return_tensors": return_tensors,
                        },
                    )

            # Combine results
            all_results = [None] * len(texts)

            # Add cached results
            for i, result in cached_results:
                all_results[i] = self._format_result(result, return_tensors)

            # Add uncached results
            for i, result in zip(uncached_indices, uncached_results):
                all_results[i] = result

            return all_results
        else:
            # No caching, use standard batch processing
            return self.tokenizer.encode(
                texts,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs,
            )

    def _format_result(
        self, result: Dict[str, Any], return_tensors: Optional[str]
    ) -> Any:
        """Format cached result according to return_tensors parameter."""
        tokens = result["tokens"]

        if return_tensors is None:
            return tokens
        elif return_tensors == "pt":
            if isinstance(tokens, list):
                return torch.tensor(tokens, device=self.device)
            return tokens
        else:
            return tokens

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Optimized decoding with batching support.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding parameters

        Returns:
            Decoded text
        """
        # Handle single sequence
        if isinstance(token_ids, (list, torch.Tensor)) and len(token_ids.shape) == 1:
            return self.tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens, **kwargs
            )

        # Handle batch of sequences
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def batch_encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> List[Union[torch.Tensor, List[int]]]:
        """
        Encode texts in batches for memory efficiency.

        Args:
            texts: List of texts to encode
            batch_size: Batch size (uses default if None)
            add_special_tokens: Whether to add special tokens
            return_tensors: Format to return tensors in
            **kwargs: Additional tokenization parameters

        Returns:
            List of encoded texts
        """
        if batch_size is None:
            batch_size = self.batch_size

        results: list = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = self._encode_batch(
                batch_texts,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs,
            )
            results.extend(batch_results)

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get tokenizer cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "caching_enabled": self.enable_caching,
        }

    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Tokenizer cache cleared")

    def optimize_for_device(self) -> None:
        """Optimize tokenizer for current device."""
        # Move tokenizer to device if it supports it
        if hasattr(self.tokenizer, "to"):
            try:
                self.tokenizer = self.tokenizer.to(self.device)
                logger.info(f"Moved tokenizer to {self.device}")
            except Exception as e:
                logger.warning(f"Failed to move tokenizer to {self.device}: {e}")

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get tokenizer optimization information."""
        return {
            "device": self.device,
            "cache_size": self.cache_size,
            "batch_size": self.batch_size,
            "caching_enabled": self.enable_caching,
            "cache_stats": self.get_cache_stats(),
        }


def create_optimized_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    cache_size: int = 1000,
    batch_size: int = 32,
    enable_caching: bool = True,
    device: str = "auto",
) -> OptimizedTokenizer:
    """
    Create an OptimizedTokenizer instance.

    Args:
        tokenizer: Hugging Face tokenizer
        cache_size: Maximum number of cached tokenizations
        batch_size: Default batch size for batched operations
        enable_caching: Whether to enable tokenization caching
        device: Device to run on

    Returns:
        Configured OptimizedTokenizer instance
    """
    return OptimizedTokenizer(
        tokenizer=tokenizer,
        cache_size=cache_size,
        batch_size=batch_size,
        enable_caching=enable_caching,
        device=device,
    )
