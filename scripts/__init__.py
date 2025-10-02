"""
Utility Scripts Module

This module contains utility scripts for common tasks such as setup,
deployment, monitoring, and maintenance.

Scripts:
- setup.py: Environment setup and dependency installation
- deploy.py: Deployment automation
- monitor.py: System monitoring and health checks
- benchmark.py: Automated benchmarking
- cleanup.py: Resource cleanup and maintenance
"""

# Script utilities and helpers
import sys
import os
from pathlib import Path

# Add src to Python path for script execution
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
