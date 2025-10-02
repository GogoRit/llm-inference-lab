"""
Ping Client for vLLM API Server

OpenAI-compatible HTTP client for testing and benchmarking vLLM servers.
Supports configurable endpoints, models, and request parameters.

Usage:
    python -m src.server.ping_vllm --host 127.0.0.1 --port 8000 \\
        --model llama-2-7b --prompt "Hello"
    python -m src.server.ping_vllm --config configs/vllm.yaml --prompt "Test prompt"
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import requests
import yaml


class VLLMPingClient:
    """OpenAI-compatible HTTP client for vLLM servers."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8000, config_path: str = None
    ):
        """
        Initialize the vLLM ping client.

        Args:
            host: Server hostname or IP address
            port: Server port number
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Override with constructor parameters if provided
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"

        # Default request parameters
        self.default_params = {
            "model": self.config.get("model", "llama-2-7b"),
            "max_tokens": self.config.get("max_tokens", 50),
            "temperature": self.config.get("temperature", 0.7),
            "stream": False,
        }

        self.logger.info(f"Initialized vLLM client for {self.base_url}")

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "model": "llama-2-7b",
            "max_tokens": 50,
            "temperature": 0.7,
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 1.0,
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded vLLM config from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default vLLM configuration")
        else:
            self.logger.info("Using default vLLM configuration")

        return default_config

    def ping(self, timeout: int = None) -> bool:
        """
        Test if the vLLM server is reachable.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if server is reachable, False otherwise
        """
        if timeout is None:
            timeout = self.config.get("timeout", 30)

        try:
            response = requests.get(f"{self.base_url}/health", timeout=timeout)
            if response.status_code == 200:
                self.logger.info("vLLM server is reachable")
                return True
            else:
                self.logger.warning(f"Server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to ping server: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        model: str = None,
    ) -> Dict[str, Any]:
        """
        Generate text using the vLLM server.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            model: Model name (uses config default if None)

        Returns:
            Dictionary containing response data and timing information
        """
        start_time = time.time()

        # Prepare request parameters
        params = self.default_params.copy()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if model is not None:
            params["model"] = model

        # Prepare request payload
        payload = {"messages": [{"role": "user", "content": prompt}], **params}

        self.logger.debug(f"Sending request to {self.chat_endpoint}")
        self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Send request with retry logic
        timeout = self.config.get("timeout", 30)
        retry_attempts = self.config.get("retry_attempts", 3)
        retry_delay = self.config.get("retry_delay", 1.0)

        for attempt in range(retry_attempts):
            try:
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    result = response.json()
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000

                    # Extract generated text
                    generated_text = ""
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            generated_text = choice["message"]["content"]

                    # Calculate tokens (approximate)
                    tokens_generated = (
                        len(generated_text.split()) if generated_text else 0
                    )

                    return {
                        "success": True,
                        "generated_text": generated_text,
                        "latency_ms": latency_ms,
                        "tokens_generated": tokens_generated,
                        "model": params["model"],
                        "prompt": prompt,
                        "raw_response": result,
                    }
                else:
                    self.logger.warning(
                        f"Server returned status {response.status_code}: "
                        f"{response.text}"
                    )
                    if attempt < retry_attempts - 1:
                        self.logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "latency_ms": (time.time() - start_time) * 1000,
                        }

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retry_attempts - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "latency_ms": (time.time() - start_time) * 1000,
                    }

        return {
            "success": False,
            "error": "All retry attempts failed",
            "latency_ms": (time.time() - start_time) * 1000,
        }

    def get_models(self) -> Dict[str, Any]:
        """
        Get available models from the server.

        Returns:
            Dictionary containing available models or error information
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                return {"success": True, "models": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="vLLM Ping Client")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server hostname or IP address"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port number")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--ping-only",
        action="store_true",
        help="Only test server connectivity, don't generate text",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize client
    client = VLLMPingClient(host=args.host, port=args.port, config_path=args.config)

    # Test connectivity
    if not client.ping():
        client.logger.error("Server is not reachable. Exiting.")
        return

    if args.ping_only:
        client.logger.info("Server connectivity test passed")
        return

    # Generate text
    result = client.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model=args.model,
    )

    # Log results
    if result["success"]:
        client.logger.info("=== vLLM Generation Results ===")
        client.logger.info(f"Model: {result['model']}")
        client.logger.info(f"Latency: {result['latency_ms']:.2f} ms")
        client.logger.info(f"Tokens generated: {result['tokens_generated']}")
        client.logger.info(f"Prompt: {result['prompt']}")
        client.logger.info(f"Generated: {result['generated_text']}")
    else:
        client.logger.error(f"Generation failed: {result['error']}")


if __name__ == "__main__":
    main()
