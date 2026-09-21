# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

`laya` is a Python package for **non-autoregressive "System 1" decision models**. It answers
typed questions (`choice`, `score`, `noul`) about a state (text, dict, or JSON document) in a
single encoder forward pass — no text generation, no parsing, no hallucination — and returns
calibrated probabilities. Weights live on the Hugging Face Hub (`convaiinnovations/laya`, plus
its `multilingual` and `typed-decisions` subfolders); this repo is the **inference runtime and
router** only. Training happens in the Kaggle notebook under `notebooks/`.

- Package name: `laya` (PyPI), version in `pyproject.toml` (`0.3.5`).
- License: Apache-2.0. Upstream: `NandhaKishorM/laya`. This checkout's remote is
  `ServaboFidem/laya-onnx` — a fork. **There is currently no ONNX code in the tree**; the name
  reflects intent, not existing content. Don't assume an ONNX export path exists.
- Python floor is **3.10**, and it is enforced by a test (`tests/test_packaging.py`, issue #34).

## Layout

```
laya/            the package (7 modules, ~1.9k lines total)
  agent.py       Agent/RLAgent runtime + laya.load(); tokenizer fixes, device resolution
  common.py      DecisionModel (nn.Module), sequence construction, rewards, ECE, temperature
  router.py      Router: picks a checkpoint per request; LRU model cache, preload, thread-safe
  lang.py        dependency-free script + Latin-language detection (routing's primary signal)
  presets.py     ready-made question schemas (triage, email, guard, moderation, router)
  email.py       email body cleaning + email_state() helper
  shortlist.py   opt-in embedding shortlist for high-cardinality choice questions
tests/           plain scripts, not pytest (see Testing)
docs/            architecture + router-flow diagrams, self-contained HTML (see Diagrams)
research/        benchmark harnesses + raw result JSON; never imported by the package
notebooks/       fine-tuning notebook (Kaggle 2xT4)
assets/          logos and benchmark plots referenced by the README
graft/           generated repo map (gitignored, but greppable via .ignore)
.github/         ci.yml, release.yml, security.yml
```

`laya/__init__.py` is the public API contract: everything exported there is what users import.
`__all__` is explicit — **if you add a public symbol, add it to both the imports and `__all__`**,
because CI runs `python -c "import laya; print(sorted(laya.__all__))"` as an import check.

## Core concepts

**Question types** (`QTYPES` in `common.py`): `choice` (label + per-option probabilities +
confidence), `score` (expected level on an ordinal rubric), `noul` (calibrated P(true)).

**The three checkpoints**, and why the router exists:

| name | repo / subfolder | encoder | context | for |
|---|---|---|---|---|
| `english` | `convaiinnovations/laya` | ModernBERT-large 421M | 512 | English |
| `multilingual` | `.../laya` + `multilingual` | mmBERT-base 322M | 1024 | 100+ languages |
| `typed-decisions` | `.../laya` + `typed-decisions` | ModernBERT-large 421M | 1024 | typed-decisions workflows |

The English checkpoint does not degrade gracefully off English — it **collapses while staying
confident** (Khmer: 0.000 accuracy at 0.952 mean confidence). Confidence gating cannot catch
this, so routing must happen *before* the forward pass, from script detection in `lang.py`.
Preserve that invariant in any change to routing.

**Token budget.** A sequence splits into an option-prompt budget (`head_max_len`) and the rest
for the state (`max_len - head_max_len`). Large label sets starve per-option tokens; that is the
known weakness (Banking77 0.425) and the reason `shortlist.py` exists. `predict` / `system_one`
still score every criterion they are given — shortlisting is opt-in and never implicit.

**`Agent.predict` is an alias for `Agent.system_one`** (`laya/agent.py:370`). Change one and you
change both.

## Working conventions

- **Stdlib-first.** `lang.py` and `email.py` are deliberately dependency-free (regex + Unicode
  ranges). Runtime deps are only torch, transformers, safetensors, huggingface_hub, numpy. Don't
  add dependencies casually; `tests/test_packaging.py` checks metadata against reality.
- **Comments explain *why*, at length.** Module docstrings carry the measured evidence behind a
  design decision (see `router.py`, `lang.py`). Match that density — a change with no rationale
  is out of style here.
- **Claims are measured or attributed.** Numbers in docstrings and the README come from
  `research/results/*.json`. Third-party figures (e.g. Jev) are labelled as published, not
  measured here. Do not invent or round benchmark numbers.
- `USE_TF=0` / `USE_TORCH=1` are set everywhere (CI, tests, research scripts): `transformers`
  probes for TensorFlow at import and its abseil runtime can deadlock model construction. Keep
  those env guards when adding an entry point.
- Lint line length is 120.
- Model lifecycle in `Router` is guarded by an `RLock`; inference is deliberately left *outside*
  the lock so concurrent predictions share a checkpoint (fix for #95). Don't widen the lock.
- Hub downloads use `allow_patterns` so a bundled repo doesn't pull sibling checkpoints.

## Testing

Tests are **standalone scripts, not pytest**. Each collects `PASS`/`FAIL` lists, prints a
summary, and calls `sys.exit(1)` on failure. They insert the repo root on `sys.path` themselves.

```bash
python tests/test_router.py          # routing + language detection (pure, no weights)
python tests/test_criteria.py        # criteria rendering (structured values -> JSON, PR #2)
python tests/test_download.py        # checkpoint download paths, tiny local weights
python tests/test_shortlist.py       # shortlist with a fake embed_fn, mocked predict
python tests/test_decision_model.py  # DecisionModel.forward on a tiny from-config BERT
python tests/test_packaging.py       # pyproject metadata vs. real dependency floors
python tests/test_email.py           # email body cleaning regressions
```

All seven run offline and are wired into both `ci.yml` and `release.yml` — **a new test file
must be added to both workflows**, or it never runs.

`tests/test_local_e2e.py` is the exception: it needs real weights on disk (`~/laya_models` by
default, or pass a root as `argv[1]`; `LAYA_DEVICE` selects the device) and is **not** in CI.
Run it manually when touching the forward path.

Lint locally the way CI does:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

## Diagrams

`docs/` holds two self-contained HTML pages — inline SVG, no build step, no JavaScript. Open them
in a browser; there is nothing to compile and nothing to install.

| file | what it shows | goes stale when |
|---|---|---|
| `docs/architecture.html` | the request path: state + questions → script detection → Router → Agent → DecisionModel → typed answers, with the Hub external and the shortlist optional | a module moves in or out of the request path |
| `docs/router-decision-flow.html` | `Router.route()`'s real precedence: explicit `model=`/`task=`/`lang=`, then opt-in workflow match, then script and English detection | the precedence chain in `laya/router.py` changes |

The second one encodes the actual branch structure of `route()`, including the `auto_task_detection`
opt-in and the no-letters fall-back to the default. **Change that function and the diagram is wrong** —
it is documentation of behaviour, not decoration.

Both were produced with the vendored `diagram-design` skill (`.claude/skills/diagram-design/`). To
regenerate or add one, invoke that skill; it owns the layout grammar and the taste gate. Two rules
for anyone touching it:

- **Don't edit the vendored skill.** It is pinned byte-for-byte to upstream `dc1ace4` so it stays
  updatable. The skin lives in a *profile* instead, which is why `style-guide.md` is untouched.
- **Verify before committing a diagram**, the way the skill asks:
  `python .claude/skills/diagram-design/scripts/self_check.py docs/<file>.html`

### The brand profile is not in this repo

`.diagram-design` (repo root) contains `profile: laya` and selects a style profile stored at
`~/.diagram-design/profiles/laya.md` — outside the repo, because an install directory can be
replaced by an update. **A fresh clone therefore has the marker but not the profile**, and the skill
will say the `laya` slug is missing rather than silently falling back. Recreate it by re-running the
skill's onboarding with these tokens, all derived from `assets/logo-*.svg`:

| role | value | source |
|---|---|---|
| `accent` | `#2a78d6` | the logo blue |
| `ink` | `#111111` | logo near-black |
| `paper` | `#eceef1` | logo off-white |
| `link` | `#1d3f6e` | derived — pushed to deep navy because `accent` already holds the blue |

Typography stays at the skill's shipped default: the logo ships no webfont (it falls back to DejaVu
Sans), so no brand family is claimed.

## Release

Tag `vX.Y.Z` and push. `release.yml` runs the tests, builds, **verifies that `pyproject.toml`'s
version equals the tag**, publishes to PyPI via trusted publishing (no token in secrets), and
opens a GitHub Release. So a version bump is a `pyproject.toml` edit *and* a matching
`laya/__init__.py::__version__` edit — keep the two in sync.

`security.yml` runs gitleaks twice (diff + whole working tree) plus a dependency CVE scan,
weekly and on every push. `.gitignore` aggressively excludes session transcripts and token
files because a PyPI token once survived in `main` inside a CLI transcript. **Never commit
anything resembling a credential or a pasted terminal log.**

## Gotchas

- `graft/` is gitignored but re-admitted to ripgrep by `.ignore`. It is a *generated* repo map —
  grep it to orient quickly, but never edit it by hand and never treat it as the source of truth.
- The README is long and doubles as the project's evidence document, including an "Honest
  limits" section. If you change behaviour that a README number depends on, update the README.
- `research/` and `notebooks/` are excluded from packaging (`packages = ["laya"]` in
  `pyproject.toml`, because root-level directories break setuptools auto-discovery).
- Both checkpoints ship over-confident, and `laya-multilingual` ships with **no fitted
  temperatures at all**. Don't write code or docs that assume out-of-the-box calibration.
