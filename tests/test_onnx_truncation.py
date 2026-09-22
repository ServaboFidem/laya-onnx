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
    # truncation_report needs the same q and head_max_len that produced `seq`, because the
    # head (instructions + options) is rebuilt with an empty state to measure its length --
    # see laya_onnx/truncation.py for why that rebuild is the correct way to isolate it.
    return truncation_report(TOK, state, seq, max_len, Q, head_max_len)


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
check("long/dropped-is-positive", r["dropped"] > 2500, True)
check("long/used-plus-dropped", r["state_tokens_used"] + r["dropped"], r["state_tokens"])

# The budget claim itself: mmBERT's 1024/256 must admit meaningfully more than 512/192.
wide = report_for(long_state, 1024, 256)["state_tokens_used"]
narrow = report_for(long_state, 512, 192)["state_tokens_used"]
check("budget/mmbert-admits-more", wide > narrow, True)
check("budget/roughly-double", wide > 2 * narrow - 64, True)

# Structured states are counted after serialization, the way the model sees them.
code = {"code": "def handler(req):\n    return {'status': 200}"}
check("code/counts-serialized", state_token_count(TOK, code) > 0, True)
check("code/short-enough-not-truncated", report_for(code, 1024, 256)["truncated"], False)

# A long structured (JSON) state must also truncate -- not just long plain-text states.
long_code = {"trace": [{"frame": i, "fn": "handler_%d" % i, "file": "a.py"} for i in range(400)]}
r = report_for(long_code, 1024, 256)
check("long_code/truncated", r["truncated"], True)
check("long_code/used-plus-dropped", r["state_tokens_used"] + r["dropped"], r["state_tokens"])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all truncation tests passed")
sys.exit(1 if FAIL else 0)
