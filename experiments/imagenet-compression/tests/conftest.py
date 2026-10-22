"""Import the file-based workflow without treating its hyphenated directory as a package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
