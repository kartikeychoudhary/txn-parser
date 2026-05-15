"""Pytest config — put scripts/ on sys.path so tests can use the same
flat-import style as the scripts themselves (`from amount_parser import ...`)."""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
