"""Root runner proxy for scripts/synthesize.py."""

import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "Handrwitten" / "scripts" / "synthesize.py"
sys.path.insert(0, str(target.parent.parent))

if __name__ == "__main__":
    runpy.run_path(str(target), run_name="__main__")
