# laya ONNX Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run laya's `multilingual` checkpoint (mmBERT-base) on commodity CPU hardware with no PyTorch, by exporting `DecisionModel` to ONNX and reimplementing the surrounding pipeline in NumPy against `onnxruntime`. Target workload: English prose, source code, JSON and logs.

**Architecture:** `DecisionModel.forward` is exported as a single ONNX graph. Everything either side of it is pure Python and gets vendored into a new top-level `laya_onnx/` package: sequence construction ahead of the graph, temperature scaling and answer formatting after it. `laya/` is never modified, so upstream syncs stay clean.

**Tech Stack:** Python >= 3.10, onnxruntime, onnx, tokenizers, numpy. Export-time only: torch, transformers.

**Spec:** [`../specs/2026-09-21-laya-onnx-port.md`](../specs/2026-09-21-laya-onnx-port.md)

## Global Constraints

- **Do not modify anything under `laya/`.** This repo is a fork tracking `NandhaKishorM/laya`; edits there conflict on every upstream sync. Vendor instead.
- **The runtime path must not import `torch` or `transformers`.** Only `laya_onnx/export/` and `laya_onnx/bench/` may. Task 7 enforces this with a test.
- Python floor is **3.10** (`pyproject.toml`), enforced by `tests/test_packaging.py`.
- **Tests are standalone scripts, not pytest.** Repo convention: collect `PASS`/`FAIL` lists, print a summary, `sys.exit(1)` on failure. Run as `python tests/test_x.py`. Do not add pytest.
- **Every new test file must be added to both `.github/workflows/ci.yml` and `.github/workflows/release.yml`**, or it never runs. Weight-dependent tests are the exception and stay out of CI (see `tests/test_local_e2e.py` for the precedent).
- Set `USE_TF=0`, `USE_TORCH=1`, `TOKENIZERS_PARALLELISM=false` in any entry point that imports transformers — TensorFlow's abseil runtime can deadlock model construction.
- Lint: `ruff check --select=E9,F63,F7,F82,F401,F811 --line-length=120`.
- **Checkpoint: `multilingual` (mmBERT-base), `max_len=1024`, `head_max_len=256`** — a ~768-token state budget against the English checkpoint's ~317. Read both values from `rl_agent_config.json`; never hardcode them.
- **This checkpoint ships no fitted temperatures.** Until Task 9 fits them, `confidence` is not meaningful and the README must say so. This is not the same posture as the `english` checkpoint, which ships calibrated.
- Temperatures are clamped to `[0.5, 5.0]` (`TEMP_MIN`/`TEMP_MAX`). Preserve this: the `english` checkpoint's shipped `choice:11+` bucket of 0.1006 sharpens logits ~10x and publishes a coin flip as a certainty.
- **No script guard.** mmBERT reads 100+ languages, so there is nothing to reject. An earlier draft of this plan had one; it was removed with the checkpoint change.
- Commit after every task.

---

## File Structure

```
laya_onnx/
  __init__.py          public API: load(), OnnxAgent
  sequence.py          vendored: serialize_state, render_criterion, render_options, build_sequence
  truncation.py        how many state tokens build_sequence dropped, per question
  tokenizer.py         TokenizerAdapter over tokenizers.Tokenizer (no transformers)
  collate.py           NumPy collate (replaces torch collate_items)
  postprocess.py       temp_bucket, clamp_temperature, confidence_from_probs, build_answers
  session.py           OnnxSession: thin onnxruntime wrapper
  runtime.py           OnnxAgent: ties the pipeline together
  export/
    __init__.py
    export_fp32.py     torch -> onnx (imports torch; the runtime never does)
    quantize_int8.py   dynamic quantization
    refit_temps.py     temperature refitting against a quantized graph
  bench/
    bench_latency.py   CPU latency
    eval_ece.py        calibration measurement
tests/
  test_onnx_sequence.py      vendored sequence vs laya's, offline
  test_onnx_truncation.py    truncation accounting, incl. code and JSON states
  test_onnx_tokenizer.py     tokenizer adapter against a synthetic vocab
  test_onnx_postprocess.py   collate + answer formatting golden values
  test_onnx_export.py        tiny-model export + torch/onnx parity + session
  test_onnx_no_torch.py      runtime imports without torch
  test_onnx_local_e2e.py     real weights, manual, NOT in CI
```

`laya_onnx/` is a sibling of `laya/`, not nested. `pyproject.toml` has `packages = ["laya"]`, so it stays out of the wheel until Task 10 addresses it deliberately.

---

### Task 1: Vendored sequence builder

**Files:**
- Create: `laya_onnx/__init__.py` (empty for now), `laya_onnx/sequence.py`
- Test: `tests/test_onnx_sequence.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `serialize_state(state) -> str`, `render_criterion(value) -> str`, `render_options(q) -> list[str]`, `build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None, truncate_left=False) -> tuple[list[int], list[int]]`. `q` is the internal form `{"t": str, "ins": str, "crit": dict|list|None}`. `tok` is any object exposing `.mask_token`, `.mask_token_id`, `.cls_token_id`, `.sep_token_id` and `tok(text, add_special_tokens=False)["input_ids"]`.

- [ ] **Step 1: Write the failing test**

The test passes the *same* fake tokenizer to both implementations, so it compares logic, not vocabulary, and needs no network or weights.

```python
"""Vendored sequence builder must match laya's byte for byte.

The vendored copy in laya_onnx/sequence.py exists so the runtime path does not
import torch. That copy can drift when upstream changes build_sequence; this
test is what catches it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence as upstream_build          # noqa: E402
from laya.common import render_options as upstream_options        # noqa: E402
from laya_onnx.sequence import build_sequence, render_options     # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


class FakeTok:
    """Deterministic word-level tokenizer; no network, no vocabulary file."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + (sum(ord(c) for c in w) % 900) for w in text.split()]}


TOK = FakeTok()

QUESTIONS = [
    {"t": "choice", "ins": "Which department?",
     "crit": {"billing": "invoices and refunds", "technical": "bugs", "other": None}},
    {"t": "score", "ins": "How urgent?", "crit": ["not urgent", "soon", "critical"]},
    {"t": "noul", "ins": "Does the user threaten to cancel?", "crit": None},
    {"t": "choice", "ins": "Nested criteria render as JSON, not a Python repr.",
     "crit": {"a": {"weight": 1, "note": "structured"}, "b": [1, 2, 3]}},
]

STATES = [
    "Plain English text about a duplicate charge.",
    {"from": "user@acme.com", "subject": "Refund", "body": "Billed twice in March."},
    {"code": "def handler(req):\n    return {'status': 200}"},
    ["turn one", "turn two"],
]

for qi, q in enumerate(QUESTIONS):
    check("options/%d" % qi, render_options(q), upstream_options(q))
    for si, state in enumerate(STATES):
        check("sequence/q%d/s%d" % (qi, si),
              build_sequence(TOK, state, q, 512, 192),
              upstream_build(TOK, state, q, 512, 192))

# Truncation paths: a long state must clip identically on both sides.
long_state = "word " * 4000
for truncate_left in (False, True):
    check("truncate/left=%s" % truncate_left,
          build_sequence(TOK, long_state, QUESTIONS[0], 512, 192, truncate_left=truncate_left),
          upstream_build(TOK, long_state, QUESTIONS[0], 512, 192, truncate_left=truncate_left))

# The option-budget squeeze path: many long options force per-option clipping.
crowded = {"t": "choice", "ins": "Pick one",
           "crit": {("label_%02d" % i): ("description " * 20) for i in range(30)}}
check("budget-squeeze", build_sequence(TOK, "state", crowded, 512, 192),
      upstream_build(TOK, "state", crowded, 512, 192))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all vendored-sequence tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_onnx_sequence.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx'`

- [ ] **Step 3: Write the implementation**

Copy `serialize_state`, `render_criterion`, `render_options` and `build_sequence` verbatim from `laya/common.py:15-86` into `laya_onnx/sequence.py`. Change nothing but the imports and the docstring header. Do not "improve" the copied code — divergence is exactly what the test exists to prevent.

```python
"""Sequence construction, vendored from laya/common.py.

Vendored rather than imported: laya.common imports torch at module scope, and the
whole point of this package is a runtime with no torch. tests/test_onnx_sequence.py
asserts this copy still matches upstream exactly, so drift fails loudly.

Source: laya/common.py:15-86 (laya 0.3.5).
"""
import json
from typing import Dict, List, Optional, Union


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


# ... render_criterion, render_options, build_sequence copied verbatim from laya/common.py ...
```

Create `laya_onnx/__init__.py` as an empty file for now; Task 7 fills it in.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_sequence.py`
Expected: PASS, `all vendored-sequence tests passed`

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/__init__.py laya_onnx/sequence.py tests/test_onnx_sequence.py
git commit -m "feat(onnx): vendor the sequence builder with an upstream parity test"
```

---

### Task 2: Truncation accounting

**Files:**
- Create: `laya_onnx/truncation.py`
- Test: `tests/test_onnx_truncation.py`

**Interfaces:**
- Consumes: `laya_onnx.sequence.serialize_state`.
- Produces: `state_token_count(tok, state) -> int`, `truncation_report(tok, state, seq, max_len) -> dict` returning `{"state_tokens": int, "state_tokens_used": int, "dropped": int, "truncated": bool}`.

**Why this task exists.** The whole reason this port uses mmBERT rather than the English
checkpoint is the state budget: ~768 tokens instead of ~317. That reasoning is only worth
anything if it can be checked. `build_sequence` truncates silently (`laya/common.py:82`) — the
state is clipped to whatever room is left and no one is told. A model that cannot see the second
half of a stack trace still answers confidently about the half it saw, and nothing in the result
distinguishes that from a well-grounded answer.

This task surfaces the drop so the budget can be verified against real inputs instead of trusted.

- [ ] **Step 1: Write the failing test**

```python
"""Truncation must be reported, not silent.

The state budget is the reason this port uses mmBERT; these tests are how we check
it against real inputs rather than assuming it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_onnx.sequence import build_sequence                                  # noqa: E402
from laya_onnx.truncation import state_token_count, truncation_report          # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


class FakeTok:
    """One token per whitespace-separated word. Same contract as build_sequence expects."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + (sum(ord(c) for c in w) % 900) for w in text.split()]}


TOK = FakeTok()
Q = {"t": "noul", "ins": "is this urgent", "crit": None}


def report_for(state, max_len, head_max_len):
    seq, _ = build_sequence(TOK, state, Q, max_len, head_max_len)
    return truncation_report(TOK, state, seq, max_len)


# A short state fits: nothing dropped, nothing flagged.
short = "the customer was billed twice"
r = report_for(short, 1024, 256)
check("short/state_tokens", r["state_tokens"], 5)
check("short/dropped", r["dropped"], 0)
check("short/truncated", r["truncated"], False)

# A state larger than the budget must report the overflow, not hide it.
long_state = " ".join("word%d" % i for i in range(4000))
r = report_for(long_state, 1024, 256)
check("long/state_tokens", r["state_tokens"], 4000)
check("long/truncated", r["truncated"], True)
check("long/dropped-is-positive", r["dropped"] > 3000, True)
check("long/used-plus-dropped", r["state_tokens_used"] + r["dropped"], r["state_tokens"])

# The budget claim itself: mmBERT's 1024/256 must admit meaningfully more than 512/192.
wide = report_for(long_state, 1024, 256)["state_tokens_used"]
narrow = report_for(long_state, 512, 192)["state_tokens_used"]
check("budget/mmbert-admits-more", wide > narrow, True)
check("budget/roughly-double", wide > 2 * narrow - 64, True)

# Structured states are counted after serialization, the way the model sees them.
code = {"code": "def handler(req):
    return {'status': 200}"}
check("code/counts-serialized", state_token_count(TOK, code) > 0, True)
check("code/short-enough-not-truncated", report_for(code, 1024, 256)["truncated"], False)

print("
%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all truncation tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_onnx_truncation.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx.truncation'`

- [ ] **Step 3: Write the implementation**

```python
"""Report how much of the state build_sequence had to drop.

laya/common.py:82 clips the state to whatever room is left after the option head and
says nothing. That silence is the failure mode this module exists to remove: an answer
about a truncated stack trace looks exactly like an answer about a whole one.
"""
from typing import Any, Dict, Union

from .sequence import serialize_state


def state_token_count(tok, state: Union[str, dict, list]) -> int:
    """Tokens the state would need if nothing were clipped."""
    return len(tok(serialize_state(state), add_special_tokens=False)["input_ids"])


def truncation_report(tok, state: Union[str, dict, list], seq, max_len: int) -> Dict[str, Any]:
    """Account for the state tokens that reached the model versus the ones that did not.

    `seq` is the id list build_sequence returned for this state. The head (instructions,
    options, separators) is whatever is not state, so the state tokens that survived are
    the room that was left -- capped at what the state actually needed.
    """
    needed = state_token_count(tok, state)
    # build_sequence appends state + one trailing SEP into whatever room remains.
    head_len = len(seq) - min(needed, max(0, max_len - len(seq) + needed)) - 1
    used = max(0, min(needed, len(seq) - head_len - 1))
    dropped = max(0, needed - used)
    return {"state_tokens": needed, "state_tokens_used": used,
            "dropped": dropped, "truncated": dropped > 0}
```

If that arithmetic proves fragile against `build_sequence`'s exact layout, compute `used`
directly instead: rebuild the head with an empty state, and take `used = len(seq) - len(head_seq)`.
Correctness matters more than elegance here — the number is the whole point of the task.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_truncation.py`
Expected: PASS, `all truncation tests passed`

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/truncation.py tests/test_onnx_truncation.py
git commit -m "feat(onnx): report state truncation instead of dropping tokens silently"
```

---

### Task 3: Tokenizer adapter

**Files:**
- Create: `laya_onnx/tokenizer.py`
- Test: `tests/test_onnx_tokenizer.py`

**Interfaces:**
- Consumes: `laya_onnx.sequence.build_sequence` (for the drop-in check).
- Produces: `TokenizerAdapter(model_dir)` with `.mask_token: str`, `.mask_token_id: int`, `.cls_token_id: int`, `.sep_token_id: int`, `.pad_token_id: int`, and `__call__(text, add_special_tokens=False) -> {"input_ids": list[int]}` — exactly the surface `build_sequence` needs, so Task 1's vendored code works against it unchanged.

- [ ] **Step 1: Write the failing test**

Builds a tiny WordLevel tokenizer on disk with the `tokenizers` library, so it needs no network and no checkpoint.

```python
"""TokenizerAdapter exposes the slice of the HF tokenizer API build_sequence needs,
backed by `tokenizers` alone so the runtime never imports transformers.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer                    # noqa: E402
from tokenizers.models import WordLevel             # noqa: E402
from tokenizers.pre_tokenizers import Whitespace    # noqa: E402

from laya_onnx.sequence import build_sequence       # noqa: E402
from laya_onnx.tokenizer import TokenizerAdapter    # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def make_checkpoint(tmp):
    """Write a minimal checkpoint layout: <tmp>/tokenizer/{tokenizer.json,special_tokens_map.json}."""
    tokdir = os.path.join(tmp, "tokenizer")
    os.makedirs(tokdir)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i, w in enumerate(["billing", "refund", "charge", "question", "choice", "state", "true", "false"]):
        vocab[w] = 5 + i
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(os.path.join(tokdir, "tokenizer.json"))
    with open(os.path.join(tokdir, "special_tokens_map.json"), "w") as f:
        json.dump({"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]",
                   "sep_token": "[SEP]", "mask_token": "[MASK]"}, f)
    return tmp


tmp = make_checkpoint(tempfile.mkdtemp())
try:
    tok = TokenizerAdapter(tmp)

    check("special/mask_token", tok.mask_token, "[MASK]")
    check("special/mask_id", tok.mask_token_id, 4)
    check("special/cls_id", tok.cls_token_id, 2)
    check("special/sep_id", tok.sep_token_id, 3)
    check("special/pad_id", tok.pad_token_id, 0)

    check("encode/known", tok("billing refund")["input_ids"], [5, 6])
    check("encode/unknown-maps-to-unk", tok("zzzz")["input_ids"], [1])
    check("encode/empty", tok("")["input_ids"], [])
    # add_special_tokens=False must not inject CLS/SEP -- build_sequence adds them itself.
    check("encode/no-special-tokens", tok("billing", add_special_tokens=False)["input_ids"], [5])

    # The adapter must be a drop-in for build_sequence.
    q = {"t": "noul", "ins": "refund question", "crit": None}
    ids, markers = build_sequence(tok, "charge state", q, 64, 32)
    check("build_sequence/starts-with-cls", ids[0], 2)
    check("build_sequence/ends-with-sep", ids[-1], 3)
    check("build_sequence/two-markers", len(markers), 2)
    check("build_sequence/markers-are-mask", [ids[m] for m in markers], [4, 4])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all tokenizer-adapter tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install tokenizers && python tests/test_onnx_tokenizer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx.tokenizer'`

- [ ] **Step 3: Write the implementation**

```python
"""Minimal tokenizer adapter: the slice of the HF API that build_sequence needs.

Backed by `tokenizers` (the Rust library) rather than `transformers`, because
transformers imports torch. Special-token ids are resolved from the checkpoint's
own files, never hardcoded -- the checkpoint owns its vocabulary.
"""
import json
import os
from typing import Dict, List

from tokenizers import Tokenizer

_DEFAULTS = {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}


class TokenizerAdapter:
    def __init__(self, model_dir: str):
        tokdir = os.path.join(model_dir, "tokenizer")
        tok_path = os.path.join(tokdir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                "No tokenizer.json in %r. A laya checkpoint ships one under tokenizer/; "
                "run the export first." % tokdir
            )
        self._tok = Tokenizer.from_file(tok_path)
        names = dict(_DEFAULTS)
        smap = os.path.join(tokdir, "special_tokens_map.json")
        if os.path.exists(smap):
            with open(smap, encoding="utf-8") as f:
                loaded = json.load(f)
            for key in _DEFAULTS:
                val = loaded.get(key)
                # transformers writes either "[MASK]" or {"content": "[MASK]", ...}
                if isinstance(val, dict):
                    val = val.get("content")
                if isinstance(val, str):
                    names[key] = val

        self.mask_token = names["mask_token"]
        self.mask_token_id = self._id(names["mask_token"])
        self.cls_token_id = self._id(names["cls_token"])
        self.sep_token_id = self._id(names["sep_token"])
        self.pad_token_id = self._id(names["pad_token"])

    def _id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise ValueError("special token %r is not in this tokenizer's vocabulary" % token)
        return int(tid)

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_tokenizer.py`
Expected: PASS, `all tokenizer-adapter tests passed`

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/tokenizer.py tests/test_onnx_tokenizer.py
git commit -m "feat(onnx): tokenizers-backed adapter with no transformers dependency"
```

---

### Task 4: NumPy collate and postprocess

**Files:**
- Create: `laya_onnx/collate.py`, `laya_onnx/postprocess.py`
- Test: `tests/test_onnx_postprocess.py`

**Interfaces:**
- Consumes: `laya_onnx.sequence.render_options`.
- Produces:
  - `collate(items, pad_id) -> dict[str, np.ndarray]` with `input_ids` (int64 `[B, L]`), `attention_mask` (int64 `[B, L]`), `marker_pos` (int64 `[B, K]`), `marker_mask` (bool `[B, K]`), `qtype` (int64 `[B]`). `items` is a list of `{"ids": list[int], "markers": list[int], "qtype": int}`.
  - `QTYPES = {"choice": 0, "score": 1, "noul": 2}`, `QTYPE_NAMES`
  - `clamp_temperature(t, lo=0.5, hi=5.0) -> float`, `temp_bucket(qtype, k) -> str`, `confidence_from_probs(p, k) -> float`
  - `build_answers(question_ids, internal_questions, items, logits, act, temperature, temperature_by_options) -> dict`

- [ ] **Step 1: Write the failing test**

```python
"""Collate and postprocess must reproduce laya's answer dicts from the same logits."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_onnx.collate import collate                                        # noqa: E402
from laya_onnx.postprocess import (                                          # noqa: E402
    QTYPES,
    build_answers,
    clamp_temperature,
    temp_bucket,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# --- clamp: the shipped choice:11+ bucket of 0.1006 must be refused ----------
check("clamp/sharpening-refused", clamp_temperature(0.1006), 0.5)
check("clamp/too-soft-refused", clamp_temperature(99.0), 5.0)
check("clamp/identity", clamp_temperature(1.7), 1.7)
check("clamp/nan", clamp_temperature(float("nan")), 1.0)
check("clamp/garbage", clamp_temperature("abc"), 1.0)

# --- buckets ----------------------------------------------------------------
check("bucket/choice-2", temp_bucket(QTYPES["choice"], 2), "choice:2")
check("bucket/choice-4", temp_bucket(QTYPES["choice"], 4), "choice:3-5")
check("bucket/score-7", temp_bucket(QTYPES["score"], 7), "score:6-10")
check("bucket/noul-12", temp_bucket(QTYPES["noul"], 12), "noul:11+")

# --- collate ----------------------------------------------------------------
items = [
    {"ids": [2, 9, 9, 3], "markers": [1, 2], "qtype": 0},
    {"ids": [2, 9, 3], "markers": [1], "qtype": 2},
]
b = collate(items, pad_id=0)
check("collate/ids-shape", b["input_ids"].shape, (2, 4))
check("collate/pad-applied", int(b["input_ids"][1, 3]), 0)
check("collate/attention", b["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
check("collate/marker-mask", b["marker_mask"].tolist(), [[True, True], [True, False]])
check("collate/dtypes", (str(b["input_ids"].dtype), str(b["marker_mask"].dtype)), ("int64", "bool"))

# --- answers ----------------------------------------------------------------
qids = ["department", "urgency", "churn"]
internal = {
    "department": {"t": "choice", "ins": "?", "crit": {"billing": None, "technical": None}},
    "urgency": {"t": "score", "ins": "?", "crit": ["low", "mid", "high"]},
    "churn": {"t": "noul", "ins": "?", "crit": None},
}
ans_items = [
    {"markers": [1, 2], "qtype": 0},
    {"markers": [1, 2, 3], "qtype": 1},
    {"markers": [1, 2], "qtype": 2},
]
logits = np.array([[2.0, 0.0, -1e4], [0.0, 1.0, 2.0], [-1.0, 1.0, -1e4]], dtype=np.float32)
act = np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)

answers = build_answers(qids, internal, ans_items, logits, act,
                        temperature=[1.0, 1.0, 1.0], temperature_by_options={})

check("answers/choice-wins", answers["department"]["choice"], "billing")
check("answers/choice-probs-sum", round(sum(answers["department"]["probabilities"].values()), 3), 1.0)
check("answers/score-type", answers["urgency"]["type"], "score")
check("answers/score-in-range", 0.0 <= answers["urgency"]["score"] <= 2.0, True)
check("answers/score-legend", answers["urgency"]["legend"], {"0": "low", "1": "mid", "2": "high"})
check("answers/noul-key", "noul" in answers["churn"], True)
check("answers/noul-is-p-true", answers["churn"]["noul"],
      round(float(np.exp(1.0) / (np.exp(-1.0) + np.exp(1.0))), 4))
# act is softmaxed inside build_answers, matching laya/agent.py:318.
check("answers/act-probability", answers["churn"]["action"]["act_probability"],
      round(float(np.exp(3.0) / (np.exp(3.0) + np.exp(0.0))), 4))

# A temperature bucket must actually be applied.
hot = build_answers(["department"], internal, [ans_items[0]], logits[:1], act[:1],
                    temperature=[1.0, 1.0, 1.0], temperature_by_options={"choice:2": 5.0})
check("answers/temperature-softens",
      hot["department"]["probabilities"]["billing"] < answers["department"]["probabilities"]["billing"],
      True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all postprocess tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_onnx_postprocess.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx.collate'`

- [ ] **Step 3: Write the implementation**

`collate.py` mirrors `laya/common.py:247` with NumPy instead of torch, dropping the training-only `target`/`label`/`meta` keys:

```python
"""NumPy collate. Mirrors laya/common.py:247 without torch, minus the training keys."""
from typing import Any, Dict, List

import numpy as np


def collate(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, np.ndarray]:
    if not items:
        raise ValueError("collate() received no items")
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)

    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)

    for i, it in enumerate(items):
        seq = it["ids"]
        ids[i, : len(seq)] = seq
        att[i, : len(seq)] = 1
        k = len(it["markers"])
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True

    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "qtype": np.array([it["qtype"] for it in items], dtype=np.int64)}
```

`postprocess.py` copies `QTYPES`, `clamp_temperature`, `temp_bucket` and `confidence_from_probs` from `laya/common.py:197-242`, then lifts the per-question loop from `laya/agent.py:322-360` verbatim — it is already NumPy. Two changes only: `self.temperature_by_options` / `self.temperature` become function parameters, and the `act` softmax that `laya/agent.py:318` does in torch moves inside `build_answers`:

```python
def _softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
```

Keep the max-subtraction in the per-question softmax too — that is what keeps it numerically stable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_postprocess.py`
Expected: PASS, `all postprocess tests passed`

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/collate.py laya_onnx/postprocess.py tests/test_onnx_postprocess.py
git commit -m "feat(onnx): numpy collate and answer postprocessing"
```

---

### Task 5: fp32 export

**Files:**
- Create: `laya_onnx/export/__init__.py`, `laya_onnx/export/export_fp32.py`
- Test: `tests/test_onnx_export.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — this is the torch side.
- Produces: `export_fp32(model, out_path, opset=17) -> str`, `ExportWrapper(model)`. CLI: `python -m laya_onnx.export.export_fp32 --model-dir DIR --out DIR`.

**Two things that will silently produce a wrong graph:**

1. `DecisionModel.forward` takes a `detach_encoder` kwarg used only in training. `ExportWrapper` drops it so the traced signature is exactly the five runtime tensors.
2. `laya/common.py:115` branches on `p.size(-1) >= 2`, and `topk(2)` exists only on the true side. Tracing bakes in whichever branch the sample hits, so **the sample must have at least 2 markers**, and the runtime must reject `k < 2` rather than feed the graph an input it was never traced for (Task 7).

- [ ] **Step 1: Write the failing test**

Uses a tiny from-config BERT, matching `tests/test_decision_model.py` — fast, offline, no weights.

```python
"""Export a tiny DecisionModel and check the ONNX graph matches torch."""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import onnxruntime as ort                                   # noqa: E402
import torch                                                # noqa: E402
from transformers import AutoConfig, AutoModel              # noqa: E402

from laya.common import DecisionModel                       # noqa: E402
from laya_onnx.export.export_fp32 import export_fp32        # noqa: E402
from laya_onnx.session import OnnxSession                   # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def tiny_model():
    cfg = AutoConfig.for_model("bert", hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=128, vocab_size=256, max_position_embeddings=128)
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    m = DecisionModel(enc, head_layers=1, n_act=2)
    m.eval()
    return m


def sample(batch=2, seq=32, markers=3, vocab=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, seq), generator=g)
    att = torch.ones((batch, seq), dtype=torch.long)
    att[-1, seq // 2:] = 0                     # ragged batch: padding must be exercised
    mpos = torch.randint(1, seq // 2, (batch, markers), generator=g)
    mmask = torch.ones((batch, markers), dtype=torch.bool)
    mmask[-1, -1] = False                      # ragged markers too
    qtype = torch.zeros((batch,), dtype=torch.long)
    return ids, att, mpos, mmask, qtype


tmp = tempfile.mkdtemp()
try:
    model = tiny_model()
    path = export_fp32(model, os.path.join(tmp, "model.onnx"))
    check("export/file-exists", os.path.exists(path))

    ids, att, mpos, mmask, qtype = sample()
    with torch.no_grad():
        t_logits, t_act = model(ids, att, mpos, mmask, qtype)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o_logits, o_act = sess.run(
        ["logits", "act_logits"],
        {"input_ids": ids.numpy(), "attention_mask": att.numpy(), "marker_pos": mpos.numpy(),
         "marker_mask": mmask.numpy(), "qtype": qtype.numpy()},
    )

    d_logits = float(np.abs(t_logits.numpy() - o_logits).max())
    d_act = float(np.abs(t_act.numpy() - o_act).max())
    check("parity/logits<1e-4", d_logits < 1e-4, "max|delta|=%.3g" % d_logits)
    check("parity/act<1e-4", d_act < 1e-4, "max|delta|=%.3g" % d_act)

    # Dynamic axes: a different batch, sequence length AND marker count must all run.
    ids2, att2, mpos2, mmask2, _ = sample(batch=3, seq=48, markers=5, seed=1)
    qtype2 = torch.tensor([0, 1, 2])
    with torch.no_grad():
        t2, _ = model(ids2, att2, mpos2, mmask2, qtype2)
    o2, _ = sess.run(["logits", "act_logits"],
                     {"input_ids": ids2.numpy(), "attention_mask": att2.numpy(),
                      "marker_pos": mpos2.numpy(), "marker_mask": mmask2.numpy(),
                      "qtype": qtype2.numpy()})
    check("dynamic/shape", o2.shape == (3, 5), "got %s" % (o2.shape,))
    d2 = float(np.abs(t2.numpy() - o2).max())
    check("dynamic/parity<1e-4", d2 < 1e-4, "max|delta|=%.3g" % d2)

    # Masked-out markers must stay at the -1e4 floor, not leak a real score.
    check("masked-markers-floored", float(o_logits[-1, -1]) < -1e3, "got %.4g" % o_logits[-1, -1])

    # Task 6: the session wrapper must agree with raw onnxruntime.
    s_logits, s_act = OnnxSession(path).run(
        {"input_ids": ids.numpy(), "attention_mask": att.numpy(), "marker_pos": mpos.numpy(),
         "marker_mask": mmask.numpy(), "qtype": qtype.numpy()})
    check("session/logits-match", float(np.abs(s_logits - o_logits).max()) < 1e-6)
    check("session/act-match", float(np.abs(s_act - o_act).max()) < 1e-6)
    check("session/float32", str(s_logits.dtype) == "float32", str(s_logits.dtype))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx export tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install onnx onnxruntime && python tests/test_onnx_export.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx.export'`

(The test also imports `laya_onnx.session`, written in Task 6. Write the empty module stub now or expect this import to fail until Task 6 — either is fine, the test is the gate for both.)

- [ ] **Step 3: Write the implementation**

```python
"""Export DecisionModel to ONNX. One of only two places that may import torch."""
import argparse
import os

import torch


class ExportWrapper(torch.nn.Module):
    """Pins the traced signature to the five runtime tensors.

    DecisionModel.forward also takes `detach_encoder`, which is training-only. Leaving it in
    the traced signature invites a caller to pass it and get a graph that silently ignores it.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def export_fp32(model, out_path: str, opset: int = 17) -> str:
    """Trace `model` to ONNX at `out_path`, with batch, sequence and marker axes dynamic."""
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # At least 2 markers: laya/common.py:115 branches on p.size(-1) >= 2 and only the true
    # side calls topk(2). Tracing with 1 marker bakes in the padded branch permanently.
    batch, seq, markers = 2, 32, 3
    vocab = int(getattr(model.encoder.config, "vocab_size", 256))
    args = (
        torch.randint(1, vocab, (batch, seq)),
        torch.ones((batch, seq), dtype=torch.long),
        torch.randint(1, seq // 2, (batch, markers)),
        torch.ones((batch, markers), dtype=torch.bool),
        torch.zeros((batch,), dtype=torch.long),
    )

    with torch.no_grad():
        torch.onnx.export(
            ExportWrapper(model), args, out_path,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "markers"},
                "marker_mask": {0: "batch", 1: "markers"},
                "qtype": {0: "batch"},
                "logits": {0: "batch", 1: "markers"},
                "act_logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    return out_path
```

Add a `main()` behind `if __name__ == "__main__":` that: sets `USE_TF=0` before importing transformers; loads a checkpoint via `laya.common.build_model` plus `safetensors.torch.load_file`; sets `encoder.config.reference_compile = False` if that attribute exists (ModernBERT's compiled path does not trace); calls `export_fp32`; and copies `rl_agent_config.json` and `tokenizer/` into the output directory so the export is self-contained.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_export.py`
Expected: PASS once Task 6 lands. Export-only assertions pass now.

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/export/ tests/test_onnx_export.py
git commit -m "feat(onnx): fp32 export with dynamic batch, sequence and marker axes"
```

---

### Task 6: ONNX session wrapper

**Files:**
- Create: `laya_onnx/session.py`
- Test: `tests/test_onnx_export.py` (the `session/*` checks written in Task 5)

**Interfaces:**
- Consumes: `laya_onnx.collate.collate` output.
- Produces: `OnnxSession(model_path, threads=None)` with `.run(batch) -> tuple[np.ndarray, np.ndarray]` returning `(logits, act_logits)` as float32.

- [ ] **Step 1: Confirm the test fails**

Run: `python tests/test_onnx_export.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'laya_onnx.session'`

- [ ] **Step 2: Write the implementation**

```python
"""onnxruntime session wrapper.

Thread count is the caller's choice: onnxruntime defaults to every core, which
oversubscribes badly when the host already runs several workers.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort

_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


class OnnxSession:
    def __init__(self, model_path: str, threads: Optional[int] = None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None:
            opts.intra_op_num_threads = int(threads)
        self._sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def run(self, batch: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        logits, act = self._sess.run(["logits", "act_logits"], {k: batch[k] for k in _INPUTS})
        return np.asarray(logits, dtype=np.float32), np.asarray(act, dtype=np.float32)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python tests/test_onnx_export.py`
Expected: PASS, `all onnx export tests passed`

- [ ] **Step 4: Commit**

```bash
git add laya_onnx/session.py
git commit -m "feat(onnx): onnxruntime session wrapper"
```

---

### Task 7: Public API

**Files:**
- Create: `laya_onnx/runtime.py`
- Modify: `laya_onnx/__init__.py`
- Test: `tests/test_onnx_no_torch.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `laya_onnx.load(model_dir, threads=None) -> OnnxAgent`; `OnnxAgent.predict(state, questions) -> dict`, aliased as `system_one` to match `laya/agent.py:370`; `OnnxAgent.cfg: dict`.

- [ ] **Step 1: Write the failing test**

The load-bearing guarantee of this package is that it runs without torch. Assert it by inspecting `sys.modules` after import.

```python
"""The runtime path must not import torch or transformers.

Dropping torch (~2 GB) for onnxruntime (~50 MB) is most of the commodity-hardware
win, and one stray module-scope import silently gives it back.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("torch", "transformers", "laya")]:
    del sys.modules[mod]

import laya_onnx  # noqa: E402

leaked = sorted({m.split(".")[0] for m in sys.modules} & {"torch", "transformers", "laya"})
check("import/no-torch-or-transformers", not leaked, "leaked: %s" % leaked)
check("api/load-exists", hasattr(laya_onnx, "load"))
check("api/agent-exists", hasattr(laya_onnx, "OnnxAgent"))
check("api/truncation-exported", hasattr(laya_onnx, "truncation_report"))
check("api/predict-aliases-system_one",
      laya_onnx.OnnxAgent.predict is laya_onnx.OnnxAgent.system_one)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all no-torch tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_onnx_no_torch.py`
Expected: FAIL with `api/load-exists` / `ModuleNotFoundError: No module named 'laya_onnx.runtime'`

- [ ] **Step 3: Write the implementation**

```python
"""OnnxAgent: the torch-free equivalent of laya.Agent, multilingual checkpoint."""
import json
import os
from typing import Any, Dict, Optional, Union

from .collate import collate
from .postprocess import QTYPES, build_answers
from .sequence import build_sequence, render_options
from .truncation import truncation_report
from .session import OnnxSession
from .tokenizer import TokenizerAdapter


def _to_internal(qdef: Dict) -> Dict:
    """Mirror of laya/agent.py:255."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


class OnnxAgent:
    def __init__(self, model_dir: str, threads: Optional[int] = None):
        with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.tok = TokenizerAdapter(model_dir)
        self.session = OnnxSession(os.path.join(model_dir, "model.onnx"), threads=threads)
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})

    def system_one(self, state: Union[str, dict, list],
                   questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ids = list(questions.keys())
        max_len = self.cfg.get("max_len", 1024)
        head_max_len = self.cfg.get("head_max_len", 256)

        items, internal = [], {}
        for qid in ids:
            q = _to_internal(questions[qid])
            internal[qid] = q
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            # The graph was traced with >= 2 markers; laya/common.py:115 takes a different
            # branch below that. Refuse rather than answer from an untraced path.
            if len(markers) < 2:
                raise ValueError("question %r has %d option(s); the exported graph requires at "
                                 "least 2" % (qid, len(markers)))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        batch = collate(items, self.tok.pad_token_id)
        logits, act = self.session.run(batch)
        answers = build_answers(ids, internal, items, logits, act,
                                self.temperature, self.temperature_by_options)
        return {
            "model": "laya-onnx",
            "answers": answers,
            "usage": {"input_tokens": int(batch["attention_mask"].sum()), "output_tokens": 0,
                      "truncated": {qid: truncation_report(self.tok, state, it["ids"], max_len)
                                    for qid, it in zip(ids, items)}},
        }

    predict = system_one


def load(model_dir: str, threads: Optional[int] = None) -> OnnxAgent:
    return OnnxAgent(model_dir, threads=threads)
```

`laya_onnx/__init__.py`:

```python
"""Torch-free ONNX runtime for laya's multilingual (mmBERT-base) checkpoint."""
from .runtime import OnnxAgent, load
from .truncation import state_token_count, truncation_report

__all__ = ["OnnxAgent", "load", "truncation_report", "state_token_count"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_onnx_no_torch.py`
Expected: PASS, `all no-torch tests passed`

- [ ] **Step 5: Commit**

```bash
git add laya_onnx/runtime.py laya_onnx/__init__.py tests/test_onnx_no_torch.py
git commit -m "feat(onnx): OnnxAgent public API mirroring laya.Agent"
```

---

### Task 8: Real checkpoint export and end-to-end parity

**Files:**
- Create: `tests/test_onnx_local_e2e.py` (manual, not in CI)

**Interfaces:**
- Consumes: `laya_onnx.load`, `laya.load`.
- Produces: an exported checkpoint directory containing `model.onnx`, `rl_agent_config.json`, `tokenizer/`.

**Prerequisite — there are no weights on this machine.** `~/laya_models` does not exist.

- [ ] **Step 1: Download the multilingual checkpoint**

```bash
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('convaiinnovations/laya', allow_patterns=['multilingual/*'])
print(p)
"
```

- [ ] **Step 2: Export it**

```bash
USE_TF=0 python -m laya_onnx.export.export_fp32 --model-dir <downloaded-path> --out ~/laya_onnx_models/english
```

Expected: `model.onnx` (~1.7 GB fp32), plus the copied config and tokenizer.

- [ ] **Step 3: Write the parity test**

Follows `tests/test_local_e2e.py`'s conventions: weight paths from argv, home-directory defaults, excluded from CI.

```python
"""End-to-end parity against the torch path. Real weights, real forward passes.

Run:  python tests/test_onnx_local_e2e.py [torch_model_dir] [onnx_model_dir]
Defaults to ~/laya_models/laya and ~/laya_onnx_models/english. Not in CI.
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya              # noqa: E402
import laya_onnx         # noqa: E402

TORCH_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/laya_models/laya")
ONNX_DIR = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/laya_onnx_models/english")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


torch_agent = laya.load(TORCH_DIR)
onnx_agent = laya_onnx.load(ONNX_DIR)

STATES = [
    {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
     "body": "We were billed twice for March. Refund the duplicate or we cancel."},
    {"code": "def charge(user):\n    stripe.charge(user.card, amount)\n    stripe.charge(user.card, amount)\n"},
    "Plain prose with no structure at all.",
]
SUITES = {"triage": laya.triage_questions(), "guard": laya.guard_questions(),
          "moderation": laya.moderation_questions(), "router": laya.router_questions()}

for sname, questions in SUITES.items():
    for i, state in enumerate(STATES):
        t = torch_agent.predict(state, questions)["answers"]
        o = onnx_agent.predict(state, questions)["answers"]
        check("%s/s%d/same-question-ids" % (sname, i), set(t) == set(o))
        for qid in t:
            check("%s/s%d/%s/type" % (sname, i, qid), t[qid]["type"] == o[qid]["type"])
            if t[qid]["type"] == "choice":
                check("%s/s%d/%s/choice" % (sname, i, qid), t[qid]["choice"] == o[qid]["choice"],
                      "torch=%s onnx=%s" % (t[qid]["choice"], o[qid]["choice"]))
                dp = max(abs(t[qid]["probabilities"][k] - o[qid]["probabilities"][k])
                         for k in t[qid]["probabilities"])
                check("%s/s%d/%s/probs<2e-3" % (sname, i, qid), dp < 2e-3, "max|delta|=%.3g" % dp)
            elif t[qid]["type"] == "score":
                d = abs(t[qid]["score"] - o[qid]["score"])
                check("%s/s%d/%s/score<2e-3" % (sname, i, qid), d < 2e-3, "delta=%.3g" % d)
            else:
                d = abs(t[qid]["noul"] - o[qid]["noul"])
                check("%s/s%d/%s/noul<2e-3" % (sname, i, qid), d < 2e-3, "delta=%.3g" % d)

# mmBERT reads non-Latin scripts: this must answer, not raise.
res = onnx_agent.predict({"body": "\u092e\u0941\u091d\u0938\u0947 \u0926\u094b \u092c\u093e\u0930"},
                          SUITES["triage"])
check("multilingual/answers-devanagari", set(res["answers"]) == set(SUITES["triage"]))

# The budget claim, measured on real input rather than assumed.
report = onnx_agent.predict(STATES[1], SUITES["triage"])["usage"]["truncated"]
check("truncation/code-state-fits", not any(r["truncated"] for r in report.values()),
      "dropped: %s" % {k: v["dropped"] for k, v in report.items() if v["truncated"]})

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx end-to-end parity tests passed")
sys.exit(1 if FAIL else 0)
```

- [ ] **Step 4: Run it**

Run: `python tests/test_onnx_local_e2e.py`
Expected: PASS. If a `choice` differs, stop and diagnose — a changed argmax on fp32 means the export is wrong, not imprecise.

- [ ] **Step 5: Commit**

```bash
git add tests/test_onnx_local_e2e.py
git commit -m "test(onnx): end-to-end fp32 parity against the torch path"
```

---

### Task 9: int8 quantization and calibration

**Files:**
- Create: `laya_onnx/export/quantize_int8.py`, `laya_onnx/bench/eval_ece.py`, `laya_onnx/export/refit_temps.py`

**Interfaces:**
- Consumes: an fp32 export directory.
- Produces: `quantize(fp32_path, int8_path) -> str`; `measure_ece(agent, dataset) -> dict`; `refit(agent, dataset) -> dict[str, float]` returning a `temperature_by_options` mapping.

**This is the task the whole port turns on.** Argmax accuracy will look fine after quantization while ECE silently degrades — and calibrated probability is what laya is *for*. Do not skip the measurement because the answers look right.

- [ ] **Step 1: Quantize**

```python
"""Dynamic int8 quantization: weights to int8, activation ranges computed at runtime."""
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(fp32_path: str, int8_path: str) -> str:
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8,
                     extra_options={"MatMulConstBOnly": True})
    return int8_path
```

- [ ] **Step 2: Measure ECE on fp32, then int8**

`eval_ece.py` reuses `laya.common.ece_score` — importing torch is fine here, `bench/` is not the runtime path — over a labelled set. Reuse the held-out suites already wired up in `research/scripts/bench_apps.py`. Report per `(question type, option count)` bucket: accuracy, mean confidence, ECE, sample count.

Run fp32 first to establish the baseline, then int8. A number without its baseline is not evidence.

- [ ] **Step 3: Decide from the numbers**

- int8 ECE within ~0.01 of fp32 → ship int8, no refit.
- int8 ECE materially worse → refit (Step 4).
- int8 *argmax accuracy* materially worse → do not ship int8. Investigate per-op exclusions, or ship fp32.

- [ ] **Step 4: Refit temperatures if needed**

`refit_temps.py` fits one temperature per `temp_bucket` by minimising NLL on held-out data against the **quantized** graph, clamps each to `[0.5, 5.0]` via `clamp_temperature`, and writes them into the int8 export's `rl_agent_config.json` under `temperature_by_options`.

The clamp is not optional. The shipped `choice:11+` bucket of 0.1006 is the cautionary case: it multiplies logits ~10x, publishing a 0.24 top probability as 0.99.

- [ ] **Step 5: Record the numbers and commit**

Write the before/after table into `laya_onnx/README.md`. Unmeasured claims about calibration are exactly what this repository's README refuses to make.

```bash
git add laya_onnx/export/quantize_int8.py laya_onnx/export/refit_temps.py \
        laya_onnx/bench/eval_ece.py laya_onnx/README.md
git commit -m "feat(onnx): int8 quantization with measured calibration and temperature refit"
```

---

### Task 10: Benchmarks, docs, CI

**Files:**
- Create: `laya_onnx/bench/bench_latency.py`, `laya_onnx/README.md`
- Modify: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `AGENTS.md`, `pyproject.toml`

- [ ] **Step 1: Benchmark latency**

Measure 1, 5, 10 and 50 questions per call, fp32 and int8, against the README's torch-CPU baseline of 193-464 ms. Report p50 and p95 over at least 50 runs after 5 warmup runs — the first call includes graph optimization and is not representative.

- [ ] **Step 2: Wire the offline tests into both workflows**

Add to the `test` job in `.github/workflows/ci.yml`, and to the test step in `.github/workflows/release.yml`:

```yaml
      - name: ONNX runtime tests
        run: |
          pip install onnx onnxruntime tokenizers
          python tests/test_onnx_sequence.py
          python tests/test_onnx_truncation.py
          python tests/test_onnx_tokenizer.py
          python tests/test_onnx_postprocess.py
          python tests/test_onnx_export.py
          python tests/test_onnx_no_torch.py
```

`tests/test_onnx_local_e2e.py` stays out — it needs weights, like `tests/test_local_e2e.py`.

- [ ] **Step 3: Add the optional dependency group**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
onnx = ["onnxruntime>=1.17", "tokenizers>=0.15", "numpy>=1.20"]
```

Leave `packages = ["laya"]` alone for now — shipping `laya_onnx` in the wheel is a separate decision and changes what `tests/test_packaging.py` must assert.

- [ ] **Step 4: Run the full suite**

```bash
python tests/test_router.py && python tests/test_criteria.py && python tests/test_download.py \
  && python tests/test_shortlist.py && python tests/test_decision_model.py \
  && python tests/test_packaging.py && python tests/test_email.py \
  && python tests/test_onnx_sequence.py && python tests/test_onnx_truncation.py \
  && python tests/test_onnx_tokenizer.py && python tests/test_onnx_postprocess.py \
  && python tests/test_onnx_export.py && python tests/test_onnx_no_torch.py
ruff check laya_onnx/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya_onnx/ tests/
```

Expected: every suite passes, including `tests/test_packaging.py` after the `pyproject.toml` edit.

- [ ] **Step 5: Update AGENTS.md**

Add `laya_onnx/` to the Layout block and a section covering: the vendoring decision and its parity test, the no-torch constraint and the test that enforces it, that `laya/` must not be edited, and that the ONNX tests are wired into both workflows. Link the spec and this plan.

- [ ] **Step 6: Commit**

```bash
git add laya_onnx/bench/ laya_onnx/README.md .github/workflows/ci.yml \
        .github/workflows/release.yml AGENTS.md pyproject.toml
git commit -m "docs(onnx): benchmarks, README, CI wiring and AGENTS.md guidance"
```

---

## Self-Review

**Spec coverage**

| spec requirement | task |
|---|---|
| Export the multilingual checkpoint | 5, 8 |
| `laya_onnx` package, same result shape | 7 |
| fp32 parity | 5, 6, 8 |
| int8 quantization | 9 |
| ECE re-measured, temperatures refit | 9 |
| Code / JSON / markup in scope | 2 (explicit tests), 8 (code state) |
| Vendor, do not refactor `laya/` | 1, 2 (parity tests guard drift) |
| Truncation reported and verifiable | 2, 8 |
| No torch in the runtime path | 7 (enforced by test) |
| `k < 2` rejected rather than answered | 5 (export note), 7 (runtime check) |
| Temperatures clamped to [0.5, 5.0] | 4, 9 |
| Latency measured against the CPU baseline | 10 |

**Type consistency:** `build_sequence` keeps upstream's exact signature throughout. `collate()` produces the five keys `OnnxSession.run` consumes. `build_answers` takes `(question_ids, internal_questions, items, logits, act, temperature, temperature_by_options)` in Tasks 4 and 7 alike. `predict`/`system_one` are aliases, matching `laya/agent.py:370`. `truncation_report` is defined in Task 2, called in Task 7 and re-exported from `__init__`.

**Known gap:** Task 9's ECE evaluation depends on the labelled datasets in `research/scripts/bench_apps.py`, which download from the Hub at runtime. If those are unavailable offline, Task 9 needs a cached subset under `laya_onnx/bench/data/` — decide when you get there, not now.

**Ordering note:** Task 5's test imports `laya_onnx.session`, which Task 6 creates. The two tasks share one test file on purpose: the export is only meaningfully verified through the session wrapper that production uses. Run Task 5 and 6 back to back.
