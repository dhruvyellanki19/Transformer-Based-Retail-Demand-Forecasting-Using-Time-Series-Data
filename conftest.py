"""
conftest.py — pytest root configuration.
Adds the project root to sys.path so src/ imports work without installation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
