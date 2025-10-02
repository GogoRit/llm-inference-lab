"""
Configuration Files Module

This module contains configuration files and utilities for managing
different environments and deployment scenarios.

Components:
- Model configuration files
- Server configuration templates
- Environment-specific settings
- Deployment configurations
- Benchmark configurations
- CUDA kernel parameters
"""

# Configuration utilities
import os
from pathlib import Path

# Base configuration directory
CONFIG_DIR = Path(__file__).parent

# Environment-specific configs
ENVIRONMENTS = ["development", "staging", "production"]
