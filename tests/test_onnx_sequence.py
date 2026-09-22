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
