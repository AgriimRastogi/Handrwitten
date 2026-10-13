"""Character set: text <-> integer indices.

WHY THIS FILE EXISTS
--------------------
The model conditions on text, so every character has to become an integer
before it can index into an embedding table. That mapping is the "alphabet",
and it must be **frozen** once training starts: if the mapping shifts, every
checkpoint trained against the old one silently becomes garbage. So we build
it once from the corpus, save it to JSON, and load that same file everywhere.

WHY WE DON'T REUSE GRAVES' ALPHABET
-----------------------------------
The 2013 reference implementation hard-codes a 75-symbol list. It is missing
uppercase Q, X and Z plus most punctuation, and — this is the real problem —
it maps anything outside the list to index 0, which is *also* the padding
index. So the original model was trained to treat "Q" and "nothing at all"
as the same token. That is a silent data-corruption bug, not a design choice.

We instead derive the alphabet from the corpus and make out-of-vocabulary
characters loud: `encode()` raises unless you explicitly pass
`errors="drop"`, and `scan_corpus()` reports the full character inventory so
you can see exactly what you are training on.

INDEX CONVENTIONS
-----------------
    0  = PAD    reserved. Never a real character.
    1+ = characters, sorted by codepoint for reproducibility.

We deliberately do NOT add BOS/EOS tokens. The character sequence is the
*conditioning input*, not something the model generates, so it has no need
to predict where the text ends — a padding mask handles variable lengths.
Stroke-level end-of-sequence is a separate thing, handled in the model.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Index 0 is reserved for padding, so real characters start at 1.
PAD_INDEX = 0


@dataclass(frozen=True)
class Alphabet:
    """An immutable, serialisable character <-> index mapping.

    Frozen because an alphabet that can be mutated after construction is
    exactly how you end up with a checkpoint that no longer matches its
    own tokenizer.

    Attributes
    ----------
    chars:
        The characters in index order. ``chars[0]`` maps to index 1, because
        index 0 is PAD.
    """

    chars: tuple[str, ...]

    # Built in __post_init__ rather than stored, so there is only one source
    # of truth (``chars``) and the two dicts can never disagree with it.
    _stoi: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _itos: dict[int, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            duplicates = [c for c, n in Counter(self.chars).items() if n > 1]
            raise ValueError(f"Alphabet contains duplicate characters: {duplicates!r}")

        # object.__setattr__ because the dataclass is frozen. This is the
        # standard escape hatch for deriving cached fields in __post_init__.
        object.__setattr__(self, "_stoi", {c: i + 1 for i, c in enumerate(self.chars)})
        object.__setattr__(self, "_itos", {i + 1: c for i, c in enumerate(self.chars)})

    # ── size ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of real characters (excluding PAD)."""
        return len(self.chars)

    @property
    def vocab_size(self) -> int:
        """Size of the embedding table you need: characters + 1 for PAD.

        This is the number to pass to ``nn.Embedding(vocab_size, d_model)``.
        Off-by-one here produces an index-out-of-range crash on the first
        batch containing the final character, which is a confusing way to
        find out.
        """
        return len(self.chars) + 1

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(self, text: str, errors: str = "raise") -> list[int]:
        """Convert a string to a list of indices.

        Parameters
        ----------
        text:
            The transcription of one handwriting line.
        errors:
            ``"raise"`` (default) blows up on any character not in the
            alphabet. ``"drop"`` silently skips them. Default to raising:
            if a character is missing, you want to know at data-prep time,
            not to discover six hours into training that your model never
            saw a semicolon.

        Returns
        -------
        A list of ints in ``[1, len(self)]``. Never contains 0.
        """
        if errors not in {"raise", "drop"}:
            raise ValueError(f"errors must be 'raise' or 'drop', got {errors!r}")

        out: list[int] = []
        for ch in text:
            index = self._stoi.get(ch)
            if index is None:
                if errors == "raise":
                    raise KeyError(
                        f"Character {ch!r} (U+{ord(ch):04X}) is not in the alphabet. "
                        f"Either rebuild the alphabet with scan_corpus(), or pass "
                        f"errors='drop' if you know you want to discard it."
                    )
                continue
            out.append(index)
        return out

    def decode(self, indices: Iterable[int]) -> str:
        """Convert indices back to a string. PAD (0) is skipped.

        Mostly a debugging tool: when a batch looks wrong, decoding it back
        to text is the fastest way to see whether the bug is in the data or
        in the model.
        """
        return "".join(self._itos[i] for i in indices if i != PAD_INDEX)

    def __contains__(self, ch: str) -> bool:
        return ch in self._stoi

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write the alphabet to JSON next to your processed dataset.

        JSON rather than pickle so it stays human-readable — you should be
        able to open this file and see exactly what your model's vocabulary
        is without running any code.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_index": PAD_INDEX,
            "chars": list(self.chars),
            # Stored redundantly as a convenience for reading the file by eye;
            # load() recomputes it rather than trusting it.
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Alphabet":
        """Read an alphabet written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=tuple(payload["chars"]))


# ── construction from a corpus ────────────────────────────────────────────────


def scan_corpus(texts: Iterable[str]) -> Counter[str]:
    """Count every character across the corpus.

    Run this *before* building an alphabet and actually read the output.
    IAM-OnDB has quirks worth knowing about: stray control characters,
    inconsistent quote styles, and a handful of characters that appear
    two or three times in the entire dataset.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)
    return counts


def build_alphabet(
    texts: Iterable[str],
    min_count: int = 1,
    extra_chars: str = "",
) -> tuple[Alphabet, Counter[str]]:
    """Derive an alphabet from the corpus.

    Parameters
    ----------
    texts:
        All transcriptions in the training split. **Only the training split**
        — building the alphabet over validation/test text leaks information
        and, more practically, produces a vocabulary containing characters
        the model has never been trained to write.
    min_count:
        Drop characters appearing fewer than this many times. A character
        seen twice in 12,000 lines cannot be learned; keeping it just adds a
        never-updated embedding row. Try ``min_count=5`` and check what falls
        out before committing to it.
    extra_chars:
        Characters to force-include even if absent from the corpus. Useful
        when you know users will type something the corpus lacks, and you
        would rather map it deliberately than crash at inference.

    Returns
    -------
    (alphabet, counts)
        ``counts`` is the full inventory *before* ``min_count`` filtering,
        so you can inspect exactly what was dropped.
    """
    counts = scan_corpus(texts)
    kept = {ch for ch, n in counts.items() if n >= min_count}
    kept.update(extra_chars)

    # Sorted by codepoint: deterministic across runs and machines. Sorting by
    # frequency would look tidier but means the mapping changes whenever the
    # data changes, which quietly invalidates old checkpoints.
    return Alphabet(chars=tuple(sorted(kept))), counts


def format_inventory(counts: Counter[str], min_count: int = 1) -> str:
    """Render a character-frequency table for terminal output.

    Formatting live in its own function so both the data-prep script and
    the inspection script print the report identically.
    """
    lines = [f"{'char':>6}  {'codepoint':>9}  {'count':>7}  kept?"]
    lines.append("-" * 40)
    for ch, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        # repr() so whitespace and control characters are visible rather
        # than silently wrecking the column alignment.
        shown = repr(ch)[1:-1] or ch
        kept = "yes" if n >= min_count else "DROPPED"
        lines.append(f"{shown:>6}  U+{ord(ch):04X}  {n:>7}  {kept}")
    total = sum(counts.values())
    lines.append("-" * 40)
    lines.append(f"{len(counts)} distinct characters, {total} total")
    return "\n".join(lines)
