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
laya_onnx/      torch-free ONNX runtime for the multilingual checkpoint (see laya_onnx)
  runtime.py     OnnxAgent: the no-torch mirror of laya.Agent; predict = system_one
  session.py     onnxruntime wrapper; numpy + onnxruntime only, never torch
  sequence.py    build_sequence, VENDORED from laya/common.py -- parity-tested, do not edit freely
  collate.py     the five arrays the graph declares; padding and marker positions
  postprocess.py logits -> typed answers; its six laya.common helpers are VENDORED, parity-tested
  tokenizer.py   TokenizerAdapter over `tokenizers`, replacing transformers' AutoTokenizer
  truncation.py  truncation_report(): how much state was dropped, which torch never reports
  export/        the ONE subpackage that may import torch; fp32 export, int8, temperature refit
  bench/         latency + ECE measurement; may import torch; never shipped in a wheel.
                 Its *functions* never fetch data -- measure_ece/refit take the examples as an
                 argument -- but eval_ece's `__main__` downloads five Hub suites. That is the
                 rule bench/__init__.py states, and the one to preserve.
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

## laya_onnx

A torch-free ONNX runtime for the `multilingual` checkpoint (mmBERT-base, 1024/256 budget).
`laya_onnx.load(dir)` returns an `OnnxAgent` whose `predict` / `system_one` return the same
answer dicts as `laya.Agent` from a process holding onnxruntime and numpy and **neither torch
nor transformers**. Spec: `docs/superpowers/specs/2026-09-21-laya-onnx-port.md`. Plan:
`docs/superpowers/plans/2026-09-21-laya-onnx-port.md`. Measurements and their provenance live in
`laya_onnx/README.md`, which is the evidence document for this package the way the root README is
for `laya/`.

**`laya_onnx` ships in no wheel.** `pyproject.toml` keeps `packages = ["laya"]`, deliberately, so
a `pip install laya` gets the torch package and nothing else; `laya_onnx` is importable from a
checkout, which is what the ONNX tests, the export CLIs and the benchmarks use. The consequence
to hold in mind: `[project.optional-dependencies]` does publish `onnx` and `onnx-export` extras,
so `pip install laya[onnx]` installs onnxruntime and tokenizers and hands the user no module
that imports them. The extras are correct for a checkout and wrong for a release, and both
files say so — shipping `laya_onnx` in the wheel is a separate decision that also changes what
`tests/test_packaging.py` has to assert.

**`laya/` is not to be edited by ONNX work.** That was a hard constraint of the port and it
stays one. `laya` is the upstream package and the fork's value is that it remains a clean
downstream of `NandhaKishorM/laya`; a refactor of `laya/common.py` "so the ONNX side can import
it" trades that away for a few saved lines. The same applies to `tests/` for the `laya/` suites.

**Vendoring, and the test that makes it safe.** `laya_onnx/sequence.py` is a copy of
`laya/common.py:15-86` (laya 0.3.5), not an import, because `laya.common` imports torch at module
scope and importing it would drag torch into the one process that exists to not have it. Vendored
code rots silently, so `tests/test_onnx_sequence.py` imports *both* implementations and asserts
they produce identical output over a fixture set spanning all three qtypes, the truncating and
non-truncating paths, `truncate_left`, and an option head crowded past `head_max_len` — change
`build_sequence` upstream and that test fails rather than the two paths quietly disagreeing. It
is behavioural equality, not a source-text diff, so a refactor upstream that preserves behaviour
passes. **If you edit `laya_onnx/sequence.py` for any reason other than tracking upstream, you
have broken the thing the test is protecting.**

`sequence.py` is not the only vendored file. `laya_onnx/postprocess.py` copies `QTYPES`,
`QTYPE_NAMES`, `confidence_from_probs`, `temp_bucket`, `TEMP_MIN`/`TEMP_MAX` and
`clamp_temperature` verbatim from `laya/common.py`, for the same reason and with the same
hazard — a shifted bucket boundary or a changed entropy normalisation upstream would leave the
two paths publishing different `confidence` values for identical logits. Those six are now
parity-tested the same way, at the bottom of `tests/test_onnx_postprocess.py`: both
implementations imported, `temp_bucket` compared over every `(qtype, k)` for k in 0..39,
`confidence_from_probs` over a spread of simplex vectors for k in 1..19, and
`clamp_temperature` over its boundaries and its pathological inputs. That file therefore
imports torch, like the sequence suite; both run in the ONNX CI job, which has it.

**Two pieces of vendored-shaped code are still covered only outside CI.** `_to_internal`
(`laya_onnx/runtime.py:46-56`, copied from `laya/agent.py:255`) and the body of
`build_answers`'s per-question loop (lifted from `laya/agent.py`'s torch path) have no
upstream-parity assertion; what checks them against the real thing is
`tests/test_onnx_local_e2e.py`, which needs real weights and is **not** in CI. Change either
side and the divergence is caught by hand or not at all.

**The no-torch constraint is enforced, not documented.** `tests/test_onnx_no_torch.py` clears
`torch`, `transformers`, `laya` and `laya_onnx` out of `sys.modules`, imports `laya_onnx`, and
asserts none of the first three came back — plus a second check that installs an import hook, so
a module that imports torch inside a `try: ... except ImportError: pass` guard is caught by name
at the moment it asks, not missed because the `sys.modules` scan ran after the cleanup. This is
why `laya_onnx/__init__.py` imports only `runtime` and `truncation`: `laya_onnx.export` may import
torch (it is the export path) and `laya_onnx.bench` pulls it in transitively through
`laya.common.ece_score`, so neither is re-exported at package level. Adding a convenience
re-export of either to `__init__.py` fails that test, which is the point.

**Ship fp32.** The int8 build is reproducible and stays in the tree, but it is measurably worse
where it matters: `noul:2` accuracy 0.8833 → 0.7800 (McNemar p = 1.6e-06; pooled p = 3.1e-05),
while ECE does not clearly separate the three graphs. On the benchmark host it bought 4–23%
latency and ~4x less disk. That is not a trade worth taking for a decision model whose output is
a label. `laya_onnx/README.md` has the full study, including what was ruled out (the embedding
table) and what was never tried (per-layer exclusion, static calibration, per-channel weights) —
those are labelled hypotheses there and should stay labelled.

**An export is two files.** `export_fp32` writes `model.onnx` (~2.8 MB of graph) *and*
`model.onnx.data` (~1.29 GB of initializers); the weights exceed protobuf's message ceiling, so
external data is not optional. Copying only `model.onnx` deploys a model that cannot run. The
exporter verifies the pairing at write time; `OnnxSession.__init__` does **not**, so a missing
sidecar surfaces at serving time as a raw onnxruntime external-data error that says nothing
about the export. Known rough edge — if you improve one thing here, improve that.

**Confidence from this checkpoint is not calibrated.** `laya-multilingual` ships
`temperature_by_options: {}`. The port fitted the first temperatures it has ever had, but only
**4 of 9 reachable buckets**, on five **English** suites, with `noul:2` landing at 4.9490 on the
int8 graph — 99% of the way to the 5.0 clamp, which `rail_status` reports as `near_rail` and
which should not be read as converged. `write_temperatures` records that coverage under
`temperature_by_options_provenance`. Do not write code or docs that treat `confidence` from this
checkpoint as calibrated.

**Latency is measured, and on the one host measured it is not a speed win.** Read that scope
before quoting the sentence: the host is a dual Xeon Gold 6148 (40 cores / 80 threads, two
sockets), which is close to the opposite of the commodity hardware this port targets, and is
exactly the regime where torch's threaded MKL kernels do best and onnxruntime's default
core-wide thread pool does worst. Nobody has measured a 4-core container, and a smaller host may
well reverse the result.

On that host (onnxruntime 1.27.0, CPUExecutionProvider), fp32 ONNX is 164.5 ms p50 at 1 question
against torch's 175.4 ms measured the same way on the same machine, and **slower** above that:
1.40x at 5 questions, 1.81x at 50. The reason to take this port is the dependency surface, not
throughput. `laya_onnx/bench/bench_latency.py` produces these — nearest-rank p50/p95 over 50
runs after 5 discarded warmup runs — and `laya_onnx/README.md` carries the full tables with the
machine block. Any new latency claim gets the same scoping these do.

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

python tests/test_onnx_sequence.py      # vendored build_sequence still matches laya/common.py
python tests/test_onnx_truncation.py    # truncation_report arithmetic
python tests/test_onnx_tokenizer.py     # TokenizerAdapter vs. transformers' behaviour
python tests/test_onnx_postprocess.py   # logits -> answers, temperature clamping
python tests/test_onnx_export.py        # export guards by inspection (no weights needed)
python tests/test_onnx_runtime.py       # OnnxAgent: config-required budget, k<2 rejection
python tests/test_onnx_calibration.py   # int8 study: binning, McNemar, rail_status
python tests/test_onnx_no_torch.py      # `import laya_onnx` must not pull torch in
```

All fifteen run offline and are wired into both `ci.yml` and `release.yml` — **a new test file
must be added to both workflows**, or it never runs. The eight ONNX suites sit in one step in
each workflow, prefixed by `pip install onnx onnxruntime tokenizers`; torch and transformers
arrive with `pip install -e .`, and `onnxscript` is deliberately absent because no offline test
runs the dynamo exporter.

Two exceptions need real weights on disk and are **not** in CI. Run them by hand when touching
the forward path:

- `tests/test_local_e2e.py` — the torch path (`~/laya_models` by default, or a root as `argv[1]`;
  `LAYA_DEVICE` selects the device).
- `tests/test_onnx_local_e2e.py` — the ONNX path against torch on the same weights. This is the
  test behind the **3.719e-05** max-|delta| parity figure in `laya_onnx/README.md`; nothing in
  CI reproduces that number, so changing the forward path without running this file means the
  claim is no longer being checked by anything.

**Known pre-existing failure on Windows.** `tests/test_download.py` fails 5 cases locally:
`tests/test_download.py:52` builds expected paths with `str(p.relative_to(...))` (backslashes on
Windows) and compares them at `:95` against `as_posix()` values. It is a path-separator bug in
the test harness, not in `laya/`, and Linux CI is unaffected. Don't "fix" it as part of unrelated
work, and don't let it mask the rest — run the suites so one failure doesn't stop the run.

Lint locally the way CI does:

```bash
ruff check laya/ laya_onnx/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ laya_onnx/ tests/
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
