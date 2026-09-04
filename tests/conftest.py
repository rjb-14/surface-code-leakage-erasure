"""Pytest configuration: ensure src is on sys.path for imports."""
import sys
from pathlib import Path

# Add src directory to sys.path so surface_code_leakage_erasure can be imported
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
