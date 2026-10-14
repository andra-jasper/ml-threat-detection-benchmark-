"""Pytest bootstrap: make src/ importable without installing the package."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
