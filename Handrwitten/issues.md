# Issues

Running log of problems found while reading through the project. Newest at the
bottom. Each entry records **evidence** (a command you can re-run) so nothing
here has to be taken on faith.

Status values: `open` | `fixed` | `wontfix` | `info`

---

## 1. `git` tracks only the two stale docs — the entire real codebase is uncommitted

- **Severity:** high
- **Status:** open
- **Where:** repo root

`git ls-files` returns exactly two files:

```
GLA_PROJECT_SYNOPSIS_REPORT.pdf
Readme.md
```

Both are the *stale* documents that describe a different project (a Rust/Sciter
desktop app). Everything of value — `hws/`, `scripts/`, `tests/`,
`pyproject.toml`, and the two accurate docs `SETUP.md` and `CLAUDE.md` — is
untracked. There is no commit history for any of the actual work.

OneDrive is file sync, not version control: it will happily propagate a
truncation or a bad edit, and it gives you no diff, no blame, and no way to get
back to "the version that passed the tests".

**Fix:** commit the tree. `datas/` is already gitignored, so the working set is
small. Do this before the first training run, when the pipeline is known-good.

**Evidence:**
```bash
git ls-files
```

---

## 2. `SETUP.md` Install section states the wrong test count (36, should be 58)

- **Severity:** low
- **Status:** open
- **Where:** [SETUP.md:14](SETUP.md#L14)

The Install section says:

```bash
pytest -q                   # expect 36 passed
```

But [SETUP.md:173](SETUP.md#L173) and `CLAUDE.md` both say **58 tests**. The
document contradicts itself 160 lines apart.

58 is correct, and the origin of the stale number is clear:

| file | `def test_` count |
|---|---|
| `tests/test_pipeline.py` | 36 |
| `tests/test_tokenizer.py` | 22 |
| **total** | **58** |

36 is exactly `test_pipeline.py` alone — so line 14 was written when that was
the only test file, and never updated after `test_tokenizer.py` was added.

**Fix:** change line 14 to `# expect 58 passed`. Better, drop the number from
the Install section entirely and keep one authoritative count in the Status
section, so it can only go stale in one place.

**Evidence:**
```bash
grep -c "def test_" tests/test_pipeline.py tests/test_tokenizer.py
```

---

## 3. Tests cannot run without `pip install -e .`, but the scripts can

- **Severity:** medium
- **Status:** open
- **Where:** `tests/` (no `conftest.py`), vs `scripts/*.py`

Both scripts patch the import path so they work uninstalled:

- [scripts/inspect_data.py:41](scripts/inspect_data.py#L41)
- [scripts/prepare_data.py:62](scripts/prepare_data.py#L62)

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`tests/` has no equivalent, and there is no `conftest.py` at the repo root or in
`tests/`. Under pytest's default `prepend` import mode, `tests/` goes on
`sys.path` but the repo root does not, so `from hws.iam import ...` fails at
collection time.

Consequence: the **Verify** step of `SETUP.md` errors out for anyone who skipped
or failed the `pip install -e .` line, and the failure is a bare
`ModuleNotFoundError` that looks like a broken repo rather than a missing step.

This asymmetry may well be deliberate — running tests implies you are set up as
a developer, while the scripts should work for someone just poking at the data.
Worth an explicit decision rather than leaving it implicit.

**Fix (pick one):**
- Note in `SETUP.md` that `pip install -e .` is **required** for `pytest`, not
  merely a convenience. Cheapest, no code change.
- Or add a root `conftest.py` containing the same one-line path insert, which
  makes `pytest` work uninstalled and matches the scripts' behaviour.

**Evidence** (with `hws` uninstalled):
```bash
pip show hws        # -> Package(s) not found: hws
pytest -q           # -> 2 errors during collection, ModuleNotFoundError: No module named 'hws'
```

---

## 4. Version number is duplicated and can drift

- **Severity:** low
- **Status:** open
- **Where:** [pyproject.toml:7](pyproject.toml#L7), [hws/__init__.py:16](hws/__init__.py#L16)

```
pyproject.toml:7      version = "0.1.0"
hws/__init__.py:16    __version__ = "0.1.0"
```

Two independent string literals for one fact. They agree today; nothing keeps
them agreeing. Harmless now, but it stops being harmless the moment a version
gets written into a checkpoint file or a generated SVG's metadata, because then
`pip show hws` and the artifact can disagree about what produced it.

**Fix:** single source of truth. Keep `__version__` in `hws/__init__.py` and
have the build read it:

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "hws.__version__"}
```

**Evidence:**
```bash
grep -rn "0\.1\.0" pyproject.toml hws/__init__.py
```

---

## 5. The only two committed files are the two misleading ones

- **Severity:** medium
- **Status:** open
- **Where:** `Readme.md`, `GLA_PROJECT_SYNOPSIS_REPORT.pdf`

`CLAUDE.md` already records that these describe a different project and are not
specifications for this codebase. Logged here because of the interaction with
issue #1: since they are also the *only* tracked files, anyone who clones this
repo receives the misleading documentation and none of the accurate
documentation or code.

**Fix:** either delete them, or add a header to `Readme.md` pointing at
`SETUP.md` as the accurate status document. Resolving issue #1 removes most of
the sting.

**Evidence:**
```bash
git ls-files
```

---

## Environment note (not a defect)

`hws` is not currently installed in the active interpreter (Python 3.13), so the
`SETUP.md` Install section has not been completed in this working copy. Issue #3
is the visible symptom.

```bash
pip show hws        # Package(s) not found: hws
```
