"""Unit tests for the standalone synthesis CLI (scripts/synthesize.py)."""

from pathlib import Path

import pytest

from scripts.synthesize import main as synthesize_main, wrap_text_to_lines

DATA_DIR = Path(__file__).resolve().parent.parent / "datas" / "processed"
CONFIG_JSON = DATA_DIR / "config.json"
ALPHABET_JSON = DATA_DIR / "alphabet.json"
TOKENIZER_JSON = DATA_DIR / "tokenizer.json"


def test_wrap_text_to_lines():
    text = "Short line\nAnother line that is definitely longer than twenty characters and needs to be wrapped"
    lines = wrap_text_to_lines(text, max_chars=20)
    assert len(lines) >= 3
    assert lines[0] == "Short line"
    for line in lines:
        assert len(line) <= 25  # textwrap keeps words intact


@pytest.mark.skipif(
    not (CONFIG_JSON.exists() and ALPHABET_JSON.exists() and TOKENIZER_JSON.exists()),
    reason="Data artifacts missing on disk",
)
def test_synthesize_cli_execution(tmp_path: Path):
    out_svg = tmp_path / "test_doc.svg"

    cli_args = [
        "--text", "Hello GLA\\nSecond line test",
        "--config", str(CONFIG_JSON),
        "--alphabet", str(ALPHABET_JSON),
        "--tokenizer", str(TOKENIZER_JSON),
        "--out", str(out_svg),
        "--ruled",
        "--margin-rule",
        "--device", "cpu",
    ]

    exit_code = synthesize_main(cli_args)
    assert exit_code == 0
    assert out_svg.is_file()

    content = out_svg.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content
    # Blue ruling lines and red margin line
    assert 'stroke="#c9d6e3"' in content
    assert 'stroke="#e8a0a0"' in content
    # Vector paths
    assert "<path" in content
