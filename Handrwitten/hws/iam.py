"""Reading the IAM On-Line Handwriting Database (IAM-OnDB).

This module does exactly one job: turn the on-disk IAM files into plain Python
objects and NumPy arrays. It performs no geometric normalisation, no
resampling and no tokenisation — those live in `hws.preprocess`. Keeping
parsing separate from processing means that when you change your mind about
resampling (you will), you don't have to re-read 12,000 XML files.

DIRECTORY LAYOUT THIS MODULE EXPECTS
------------------------------------
    <root>/
      ascii-all/ascii/a01/a01-000/a01-000u.txt          transcriptions
      lineStrokes-all/lineStrokes/a01/a01-000/          pen trajectories
        a01-000u-01.xml
        a01-000u-02.xml
        ...
      original-all/original/a01/a01-000/strokesu.xml    writer IDs  <-- optional

The paths are configurable; those are just the defaults matching how the
official tarballs extract.

THE THREE FILE TYPES
--------------------
**lineStrokes XML** — one file per *line* of handwriting. Contains a
``<StrokeSet>`` of ``<Stroke>`` elements, each holding ``<Point x y time/>``.
This is the actual pen trajectory and it is what the model learns to produce.

**ascii TXT** — one file per *form* (a form is one page written by one
person). Contains two blocks:

    OCR:     the printed text the writer was asked to copy
    CSR:     what the writer actually wrote

**Only the CSR block is usable.** The writer made mistakes, and the strokes
record the mistakes, not the prompt. In ``a01-000u.txt`` the OCR block says
"nominating" while CSR says "nomnating" — the pen strokes spell the typo. If
you train against OCR text you are teaching the model that a particular set
of strokes spells a word it does not spell. The 2013 reference implementation
gets this right; it is just never explained, so it is easy to "simplify" the
parser and silently break your labels.

**original XML** — one file per form, holding
``<General><Form writerID="..."/></General>``. This is the *only* place
writer identity is recorded; it appears nowhere in the lineStrokes files.
Without it you cannot condition on writer style, which is the single biggest
improvement available over the 2013 model. This module treats it as optional
so you can start building immediately, but `writer_id` will be ``None`` until
you extract ``original-all.tar``.

COORDINATE CONVENTION
---------------------
IAM records raw whiteboard-sensor coordinates with the sensor at ``top_left``,
so **y increases downward**. We negate y at parse time so that +y is up,
matching normal mathematical convention. Every downstream module — including
the renderer — assumes +y is up. Get this wrong and your handwriting comes
out mirrored vertically, which is obvious once rendered and baffling until
then.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from xml.etree import ElementTree

import numpy as np

logger = logging.getLogger(__name__)

# Column indices into the (N, 4) point arrays this module returns. Named
# constants because `points[:, 2]` at a call site three files away is
# unreadable, and swapping two of these is a bug you will not spot by eye.
X, Y, PEN_UP, TIME = 0, 1, 2, 3
N_POINT_COLS = 4

# "a01-000u-01.xml" -> form "a01-000u", line 1
_STROKE_FILENAME_RE = re.compile(r"^(?P<form>.+)-(?P<line>\d+)$")

# "strokesu.xml" -> suffix "u";  "strokes.xml" -> suffix ""
_ORIGINAL_FILENAME_RE = re.compile(r"^strokes(?P<suffix>[a-zA-Z]*)$")


@dataclass
class RawLine:
    """One line of handwriting: the text, and the pen trajectory that wrote it.

    Attributes
    ----------
    form_id:
        e.g. ``"a01-000u"``. Identifies the page. Every line on a form was
        written by the same person, which is what makes writer-disjoint
        splitting possible.
    line_index:
        1-based index of this line within the form.
    text:
        The CSR transcription — what was actually written.
    points:
        ``(N, 4)`` float32 array of ``[x, y, pen_up, time]``.

        * ``x``, ``y`` — raw sensor units, y already flipped so +y is up.
          Not normalised; `hws.preprocess` handles that.
        * ``pen_up`` — 1.0 on the final point of each stroke, else 0.0.
          "The pen leaves the paper after this point."
        * ``time`` — seconds since the first point of this line.
    writer_id:
        Integer writer identity, or ``None`` if ``original-all`` is absent.
    source:
        Path to the stroke XML, for error messages worth acting on.
    """

    form_id: str
    line_index: int
    text: str
    points: np.ndarray
    writer_id: int | None
    source: Path

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_strokes(self) -> int:
        """Number of pen-down..pen-up segments (i.e. how many times the pen lifted)."""
        return int(self.points[:, PEN_UP].sum())

    @property
    def duration(self) -> float:
        """Wall-clock seconds the writer took to write this line."""
        if self.n_points == 0:
            return 0.0
        return float(self.points[-1, TIME])

    @property
    def time_is_monotonic(self) -> bool:
        """False if timestamps ever go backwards -- i.e. the file is corrupt.

        Roughly 0.25% of IAM lines fail this, but they fail hard: one sample has
        a -746 second step, and the violations occur *inside* strokes as well as
        between them, so it is not a stroke-ordering convention being misread.
        Time does not run backwards; those timestamps are simply wrong.

        The geometry in these files still looks fine, so this is exposed as a
        flag rather than a parse error. It matters because ``dt`` feeds pen
        velocity, and a negative dt makes velocity meaningless — which would
        silently corrupt both the ``dt`` token channel and any velocity-driven
        stroke width. :func:`iter_raw_lines` skips these by default.
        """
        if self.n_points < 2:
            return True
        return bool(np.all(np.diff(self.points[:, TIME]) >= 0))


# ── individual file parsers ───────────────────────────────────────────────────


def parse_stroke_xml(path: str | Path) -> np.ndarray:
    """Parse one lineStrokes XML into an ``(N, 4)`` array of ``[x, y, pen_up, time]``.

    Raises
    ------
    ValueError
        If the file has no ``<StrokeSet>``, or contains zero points. Both
        occur in the wild; callers are expected to skip and log rather than
        die, which is why this raises instead of returning an empty array
        that would quietly poison the dataset.
    """
    path = Path(path)

    # ElementTree honours the `<?xml encoding="ISO-8859-1"?>` declaration
    # itself, so we hand it the path rather than pre-decoded text.
    root = ElementTree.parse(path).getroot()

    stroke_set = root.find("StrokeSet")
    if stroke_set is None:
        raise ValueError(f"{path}: no <StrokeSet> element")

    rows: list[tuple[float, float, float, float]] = []
    for stroke in stroke_set.findall("Stroke"):
        points = stroke.findall("Point")
        if not points:
            # An empty <Stroke> would otherwise mark the *previous* point as
            # pen_up twice. Skip it entirely.
            continue

        last = len(points) - 1
        for i, point in enumerate(points):
            rows.append(
                (
                    float(point.attrib["x"]),
                    # Negate: sensor origin is top-left, so raw y grows
                    # downward. See module docstring.
                    -float(point.attrib["y"]),
                    # pen_up marks the END of a stroke, not the start of the
                    # next one. The distinction matters when you split on it.
                    1.0 if i == last else 0.0,
                    float(point.attrib["time"]),
                )
            )

    if not rows:
        raise ValueError(f"{path}: <StrokeSet> contained no points")

    points_array = np.asarray(rows, dtype=np.float32)

    # Make time relative to the start of this line. Absolute IAM timestamps
    # are seconds since the recording session began, so they run into the
    # thousands and differ wildly between files — useless as a model input,
    # and a silent source of huge activations if fed in raw.
    points_array[:, TIME] -= points_array[0, TIME]

    return points_array


def parse_ascii_txt(path: str | Path) -> list[str]:
    """Extract the CSR transcription lines from one ascii TXT file.

    Returns the lines in document order, so ``result[i]`` corresponds to
    stroke file ``<form>-{i+1:02d}.xml``.

    Raises
    ------
    ValueError
        If there is no ``CSR:`` marker. A file with an OCR block but no CSR
        block has no usable labels.
    """
    path = Path(path)

    # latin-1 rather than utf-8: these files are ISO-8859-1 and a few contain
    # bytes in 0x80-0xFF. utf-8 would raise UnicodeDecodeError on those, and
    # errors="ignore" would silently delete characters from your labels.
    text = path.read_text(encoding="latin-1")

    lines = [line.strip() for line in text.splitlines()]

    try:
        csr_at = lines.index("CSR:")
    except ValueError as exc:
        raise ValueError(f"{path}: no 'CSR:' marker found") from exc

    # Everything after the marker; blank lines (including the one immediately
    # following "CSR:") are dropped. Slicing a fixed +2 like the reference
    # implementation does breaks on files with a different number of blank
    # lines, and there are a few.
    return [line for line in lines[csr_at + 1 :] if line]


def load_writer_ids(original_root: str | Path | None) -> dict[str, int]:
    """Build a ``{form_id: writer_id}`` map from the ``original`` tree.

    Returns an empty dict — with a warning — if the directory is missing.
    This is deliberate: you can build and test the whole pipeline without
    writer IDs, then re-run once ``original-all.tar`` is extracted and get
    style conditioning without touching any other code.

    Notes
    -----
    Writer identity is per-*form*, not per-line: one person wrote every line
    on a page. The relevant element looks like::

        <General>
          <Form id="a01-000z" writerID="10026" saveTime="12925045.62"/>
        </General>

    We read ``id`` straight off the ``<Form>`` element rather than deriving
    it from the file path. Both usually agree, but the file is authoritative
    and this way a naming irregularity produces a correct mapping instead of
    a silently wrong one. Path derivation is kept only as a fallback for
    files whose ``<Form>`` element has no ``id``.

    Coverage is incomplete in the real corpus — roughly 1,575 of 1,907 forms
    carry a writerID. Lines on the remaining forms get ``writer_id=None``.
    Decide deliberately whether to drop them or train them with a
    "writer unknown" embedding; do not let it happen by accident.
    """
    if original_root is None:
        logger.warning(
            "No original path given -> writer IDs unavailable. "
            "Style conditioning and writer-disjoint splits will not work."
        )
        return {}

    original_root = Path(original_root)
    if not original_root.is_dir():
        logger.warning(
            "original/ not found at %s -> writer IDs unavailable. "
            "Extract original-all.tar there to enable style conditioning.",
            original_root,
        )
        return {}

    writer_ids: dict[str, int] = {}
    n_no_writer_field = 0
    n_unreadable = 0

    for xml_path in sorted(original_root.rglob("strokes*.xml")):
        # These files are ~300 KB each because they embed the whole StrokeSet,
        # but <General><Form/> sits in the first few lines. A full
        # ElementTree.parse() would read all 300 KB x 1734 files (~500 MB of
        # I/O, about a minute) to extract two attributes. iterparse streams
        # and we bail the moment we have what we need — roughly 300x less I/O.
        #
        # The explicit `open` matters: passing a path to iterparse leaves the
        # file handle open when you break out of the loop mid-stream, which on
        # Windows will eventually exhaust handles across 1700+ files.
        raw_writer: str | None = None
        form_id: str | None = None

        try:
            with open(xml_path, "rb") as handle:
                for _event, elem in ElementTree.iterparse(handle, events=("start",)):
                    if elem.tag == "Form":
                        raw_writer = elem.attrib.get("writerID")
                        form_id = elem.attrib.get("id")
                        break
                    # Reaching StrokeSet means General came and went without a
                    # Form element. Stop rather than stream the whole file.
                    if elem.tag == "StrokeSet":
                        break
        except (ElementTree.ParseError, OSError):
            n_unreadable += 1
            continue

        if raw_writer is None:
            n_no_writer_field += 1
            continue

        # Prefer the id recorded in the file; fall back to reconstructing it
        # from <dirname><filename suffix>, e.g. "a01-000" + "z".
        if not form_id:
            match = _ORIGINAL_FILENAME_RE.match(xml_path.stem)
            if match is None:
                n_no_writer_field += 1
                continue
            form_id = xml_path.parent.name + match.group("suffix")

        try:
            writer_ids[form_id] = int(raw_writer)
        except ValueError:
            n_no_writer_field += 1

    logger.info(
        "Loaded %d writer IDs covering %d distinct writers "
        "(%d files had no writerID, %d unreadable)",
        len(writer_ids),
        len(set(writer_ids.values())),
        n_no_writer_field,
        n_unreadable,
    )
    return writer_ids


# ── corpus traversal ──────────────────────────────────────────────────────────


@dataclass
class ParseStats:
    """Counters describing what happened during a corpus walk.

    Every skipped file is counted. A pipeline that silently discards 30% of
    the corpus and reports nothing is worse than one that crashes, because
    you will spend a week wondering why the model underperforms.
    """

    n_forms: int = 0
    n_lines_indexed: int = 0
    n_lines_yielded: int = 0
    n_forms_no_strokes: int = 0
    n_forms_count_mismatch: int = 0
    n_lines_unparseable: int = 0
    n_lines_blacklisted: int = 0
    n_lines_no_writer: int = 0
    n_lines_bad_time: int = 0

    def summary(self) -> str:
        lines = [
            f"forms scanned         : {self.n_forms}",
            f"lines indexed         : {self.n_lines_indexed}",
        ]
        # Only meaningful when iter_raw_lines actually parsed XML. Printing
        # "parsed OK: 0" after a pure indexing pass reads as a catastrophic
        # failure when nothing is wrong.
        if self.n_lines_yielded or self.n_lines_unparseable:
            lines.append(f"lines parsed OK       : {self.n_lines_yielded}")
        lines += [
            f"  skipped, no strokes : {self.n_forms_no_strokes} forms",
            f"  skipped, count mism.: {self.n_forms_count_mismatch} forms",
            f"  skipped, bad XML    : {self.n_lines_unparseable} lines",
            f"  skipped, bad time   : {self.n_lines_bad_time} lines",
            f"  skipped, blacklist  : {self.n_lines_blacklisted} lines",
            f"  missing writer ID   : {self.n_lines_no_writer} lines",
        ]
        return "\n".join(lines)


@dataclass
class LineRef:
    """A *pointer* to one line: everything except the pen trajectory.

    Building these is cheap — it needs only directory globs and the small
    TXT files, no XML parsing. Getting the full corpus index takes about a
    second, versus a couple of minutes to parse all 12,000 stroke files.

    That distinction matters more than it sounds. It lets tools sample 20
    random lines from across the whole corpus and parse only those 20,
    instead of either parsing everything or being stuck looking at the first
    20 lines of writer a01 over and over.
    """

    form_id: str
    line_index: int
    text: str
    stroke_path: Path
    writer_id: int | None

    def load(self) -> RawLine:
        """Parse this line's stroke XML and return the full :class:`RawLine`.

        Raises whatever :func:`parse_stroke_xml` raises; callers that are
        walking the corpus should catch and count rather than die.
        """
        return RawLine(
            form_id=self.form_id,
            line_index=self.line_index,
            text=self.text,
            points=parse_stroke_xml(self.stroke_path),
            writer_id=self.writer_id,
            source=self.stroke_path,
        )


def iter_line_refs(
    ascii_root: str | Path,
    strokes_root: str | Path,
    original_root: str | Path | None = None,
    blacklist: Sequence[str] | None = None,
    stats: ParseStats | None = None,
) -> Iterator[LineRef]:
    """Index the corpus without reading any stroke XML.

    This is where all the file-pairing logic lives; :func:`iter_raw_lines`
    is a thin wrapper that adds parsing. Keeping them separate means the
    pairing rules exist in exactly one place.

    Parameters
    ----------
    ascii_root:
        Directory containing the ``a01/``, ``a02/`` ... subtrees of TXT files.
    strokes_root:
        Same, for the lineStrokes XML files.
    original_root:
        Optional; enables ``writer_id``. Pass ``None`` if not yet extracted.
    blacklist:
        Stroke filenames (e.g. ``"a01-000u-01.xml"``) to skip. IAM contains
        some genuinely unusable lines — sensor dropouts, lines written largely
        off-sensor. The reference implementation ships a ``blacklist.npy`` of
        samples its trained model scored as very unlikely; you can reuse it or
        build your own later. Not a prerequisite for a first model.
    stats:
        Optional :class:`ParseStats` to accumulate into.

    Yields
    ------
    LineRef
        In sorted form order, so runs are reproducible.
    """
    ascii_root = Path(ascii_root)
    strokes_root = Path(strokes_root)
    stats = stats if stats is not None else ParseStats()
    blacklist_set = set(blacklist or ())

    writer_ids = load_writer_ids(original_root)

    # Sorted for determinism: filesystem iteration order varies between
    # machines, and a dataset whose contents depend on that is not reproducible.
    for txt_path in sorted(ascii_root.rglob("*.txt")):
        stats.n_forms += 1

        form_id = txt_path.stem  # "a01-000u"

        try:
            texts = parse_ascii_txt(txt_path)
        except (ValueError, OSError) as exc:
            logger.debug("Skipping form %s: %s", form_id, exc)
            stats.n_forms_no_strokes += 1
            continue

        # lineStrokes mirrors the ascii tree structure, so we can locate the
        # stroke directory by swapping the root and reusing the relative path.
        stroke_dir = strokes_root / txt_path.parent.relative_to(ascii_root)
        if not stroke_dir.is_dir():
            stats.n_forms_no_strokes += 1
            continue

        # Files for this form are "<form_id>-NN.xml". The sibling form
        # "a01-000x" lives in the same directory, hence the exact prefix.
        stroke_paths = sorted(stroke_dir.glob(f"{form_id}-*.xml"))
        if not stroke_paths:
            stats.n_forms_no_strokes += 1
            continue

        # A mismatch means we cannot trust *any* pairing on this form: if
        # there are 6 stroke files and 5 text lines, we do not know which
        # line is missing, so every pairing from that point on is wrong.
        # Dropping the whole form is the only safe option. (The reference
        # implementation asserts here and crashes, then special-cases the
        # one file it happened to notice.)
        if len(stroke_paths) != len(texts):
            logger.debug(
                "Form %s: %d stroke files vs %d text lines -> skipping form",
                form_id,
                len(stroke_paths),
                len(texts),
            )
            stats.n_forms_count_mismatch += 1
            continue

        writer_id = writer_ids.get(form_id)
        if writer_id is None and writer_ids:
            # Only counted as "missing" when we actually loaded writer data;
            # otherwise every line would be flagged, which is just noise.
            stats.n_lines_no_writer += len(stroke_paths)

        for stroke_path, text in zip(stroke_paths, texts):
            if stroke_path.name in blacklist_set:
                stats.n_lines_blacklisted += 1
                continue

            match = _STROKE_FILENAME_RE.match(stroke_path.stem)
            line_index = int(match.group("line")) if match else 0

            stats.n_lines_indexed += 1
            yield LineRef(
                form_id=form_id,
                line_index=line_index,
                text=text,
                stroke_path=stroke_path,
                writer_id=writer_id,
            )


def iter_raw_lines(
    ascii_root: str | Path,
    strokes_root: str | Path,
    original_root: str | Path | None = None,
    blacklist: Sequence[str] | None = None,
    stats: ParseStats | None = None,
    require_monotonic_time: bool = True,
) -> Iterator[RawLine]:
    """Walk the corpus, yielding one :class:`RawLine` per line of handwriting.

    Same arguments as :func:`iter_line_refs`, plus stroke XML parsing.

    Parameters
    ----------
    require_monotonic_time:
        Skip lines whose timestamps run backwards (see
        :attr:`RawLine.time_is_monotonic`). Affects ~0.25% of the corpus. Left
        on by default because a negative ``dt`` poisons the velocity channel;
        set it False if you only care about geometry and want those ~30 lines
        back.

    Unparseable files are logged and counted, not raised.

    A generator rather than a list-returning function because the full corpus at
    full resolution is roughly 1 GB in RAM as float32. Streaming lets the caller
    decide what to keep.
    """
    stats = stats if stats is not None else ParseStats()

    for ref in iter_line_refs(
        ascii_root, strokes_root, original_root, blacklist=blacklist, stats=stats
    ):
        try:
            line = ref.load()
        except (ValueError, ElementTree.ParseError, OSError) as exc:
            logger.debug("Skipping %s: %s", ref.stroke_path.name, exc)
            stats.n_lines_unparseable += 1
            continue

        if require_monotonic_time and not line.time_is_monotonic:
            logger.debug(
                "Skipping %s: timestamps go backwards (min dt %.1fs)",
                ref.stroke_path.name,
                float(np.diff(line.points[:, TIME]).min()),
            )
            stats.n_lines_bad_time += 1
            continue

        stats.n_lines_yielded += 1
        yield line


# ── splitting ─────────────────────────────────────────────────────────────────


def split_by_writer(
    lines: Sequence[RawLine] | Sequence[LineRef],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, list]:
    """Split into train/val/test with **no writer appearing in two splits**.

    Accepts either :class:`RawLine` or :class:`LineRef` — only ``writer_id``
    is read, so you can split the cheap index before deciding what to parse.

    Why this matters more than it looks: if you split randomly by line, the
    same person's handwriting lands in both train and test. The model then
    gets to memorise that individual's letter shapes and reproduce them at
    test time, so your validation loss measures *memorisation of 221 specific
    hands* rather than the thing you care about — generalising to a hand it
    has never seen. Style-conditioning results in particular become
    meaningless, because "generalise to a new writer" is exactly the claim.

    Splitting by writer costs you a little validation loss and buys you a
    number you can actually believe.

    Raises
    ------
    ValueError
        If any line lacks a ``writer_id``. Extract ``original-all.tar`` first
        — this function refuses to silently fall back to a line-level split,
        because that failure would be invisible in the metrics.
    """
    missing = sum(1 for line in lines if line.writer_id is None)
    if missing:
        raise ValueError(
            f"{missing}/{len(lines)} lines have no writer_id, so a writer-disjoint "
            f"split is impossible. Extract original-all.tar and pass original_root "
            f"to iter_raw_lines(). Refusing to fall back to a line-level split: it "
            f"would inflate your validation scores without any visible warning."
        )

    writers = sorted({line.writer_id for line in lines})  # sorted -> reproducible
    rng = np.random.default_rng(seed)
    rng.shuffle(writers)

    n_val = max(1, int(round(len(writers) * val_frac)))
    n_test = max(1, int(round(len(writers) * test_frac)))

    val_writers = set(writers[:n_val])
    test_writers = set(writers[n_val : n_val + n_test])

    splits: dict[str, list[RawLine]] = {"train": [], "val": [], "test": []}
    for line in lines:
        if line.writer_id in val_writers:
            splits["val"].append(line)
        elif line.writer_id in test_writers:
            splits["test"].append(line)
        else:
            splits["train"].append(line)

    logger.info(
        "Split %d lines / %d writers -> train %d, val %d, test %d",
        len(lines),
        len(writers),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits
