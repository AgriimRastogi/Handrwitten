"""Root runner proxy for training the new Guided-Attention Handwriting Transformer."""

import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "Handrwitten" / "new_model" / "train.py"
sys.path.insert(0, str(target.parent.parent))

if __name__ == "__main__":
    runpy.run_path(str(target), run_name="__main__")
