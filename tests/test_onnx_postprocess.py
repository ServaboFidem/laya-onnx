"""Collate and postprocess must reproduce laya's answer dicts from the same logits.

**The vendored half is parity-tested, like sequence.py's.** `postprocess.py` copies `QTYPES`,
`QTYPE_NAMES`, `confidence_from_probs`, `temp_bucket`, `TEMP_MIN`/`TEMP_MAX` and
`clamp_temperature` verbatim from `laya/common.py` (see that module's docstring for the exact
line numbers), because importing `laya.common` would drag torch into the one process that
exists not to have it. Vendored code rots silently: an upstream shift of a bucket boundary, or
a change to the entropy normalisation in `confidence_from_probs`, would leave the ONNX path
publishing a different `confidence` and a different temperature than the torch path for the
same logits, and every assertion below against a hardcoded literal would keep passing.
`tests/test_onnx_sequence.py` solved this for `build_sequence` by importing *both*
implementations and asserting behavioural equality; the section at the end of this file does
the same for these six. Like that suite, it imports `laya.common` and therefore torch -- which
is fine here (this is a test, not the runtime path) and is enforced separately by
`tests/test_onnx_no_torch.py`.
"""
import math
import os
import sys

# laya.common imports torch, and transformers probes for TensorFlow at import; every
# torch-touching entry point in this repo sets these (CI sets them job-wide, a local run of
# this file does not). The parity section at the bottom is why this file is one of them.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya.common as upstream                                               # noqa: E402
from laya_onnx.collate import collate                                        # noqa: E402
import laya_onnx.postprocess as vendored                                     # noqa: E402
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

# --- answers ------------------------------------------------------------------
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

# --- upstream parity: the vendored helpers must still behave like laya.common ---------------
# Behavioural equality, not a source diff, exactly as tests/test_onnx_sequence.py does it: a
# refactor upstream that preserves behaviour passes, a changed bucket boundary or a changed
# entropy normalisation fails here instead of silently shipping two different confidences.
check("parity/QTYPES", vendored.QTYPES, upstream.QTYPES)
check("parity/QTYPE_NAMES", vendored.QTYPE_NAMES, upstream.QTYPE_NAMES)
check("parity/TEMP_MIN", vendored.TEMP_MIN, upstream.TEMP_MIN)
check("parity/TEMP_MAX", vendored.TEMP_MAX, upstream.TEMP_MAX)

# Every (qtype, k) the bucket function can be asked for, well past the 11+ boundary. k=0 and
# k=1 are included because `temp_bucket` is total over ints and a divergence there would be
# invisible to any test that only ever passes realistic option counts.
bucket_mismatch = [
    (qt, k, vendored.temp_bucket(qt, k), upstream.temp_bucket(qt, k))
    for qt in sorted(upstream.QTYPE_NAMES)
    for k in range(40)
    if vendored.temp_bucket(qt, k) != upstream.temp_bucket(qt, k)
]
check("parity/temp_bucket-over-all-qtypes-k0..39", bucket_mismatch, [])

# A spread of simplex vectors per k: uniform (maximum entropy, confidence 0), an almost-one-hot
# (minimum entropy, confidence ~1), a geometric decay, a reversed decay, one vector padded with
# trailing mass that `p[:k]` must ignore, and seeded Dirichlet draws at three concentrations.
# Exact equality is the right assertion: the two implementations are meant to be the same
# arithmetic in the same float64, so a tolerance would only hide a real divergence.
rng = np.random.default_rng(0)
conf_mismatch = []
for k in range(1, 20):
    cases = [
        np.full(k, 1.0 / k),
        np.eye(k)[0] * 0.999 + np.full(k, 0.001 / k),
        np.array([0.5 ** (i + 1) for i in range(k)]),
        np.array([0.5 ** (k - i) for i in range(k)]),
        np.full(2 * k, 1.0 / (2 * k)),
    ]
    cases += [rng.dirichlet(np.full(k, alpha)) for alpha in (0.2, 1.0, 5.0)]
    for j, p in enumerate(cases):
        p = np.asarray(p, dtype=np.float64)
        got, want = vendored.confidence_from_probs(p, k), upstream.confidence_from_probs(p, k)
        if got != want:
            conf_mismatch.append((k, j, got, want))
check("parity/confidence_from_probs-k1..19", conf_mismatch, [])
# ...and that the shared definition is the one documented: normalized Shannon entropy. If both
# copies drifted together this is the check that still notices.
check("parity/confidence-uniform-is-zero", vendored.confidence_from_probs(np.full(4, 0.25), 4), 0.0)
check("parity/confidence-k1-is-one", vendored.confidence_from_probs(np.array([1.0]), 1), 1.0)
_half = np.array([0.5, 0.25, 0.25])
check("parity/confidence-hand-computed",
      round(vendored.confidence_from_probs(_half, 3), 10),
      round(float(1.0 - (0.5 * math.log(2) + 0.5 * math.log(4)) / math.log(3)), 10))

# The clamp, over its boundaries and its pathological inputs -- including 0.1006, the value the
# `english` checkpoint actually ships for `choice:11+` and the reason the clamp exists.
clamp_mismatch = []
for t in (0.1006, 0.0, 0.5, 0.49999, 0.50001, 1.0, 4.9490, 5.0, 5.00001, 99.0, -3.0,
          float("nan"), float("inf"), float("-inf"), "abc", None, "", "2.5", 3, True):
    got, want = vendored.clamp_temperature(t), upstream.clamp_temperature(t)
    if got != want or type(got) is not type(want):
        clamp_mismatch.append((t, got, want))
check("parity/clamp_temperature-boundaries-and-pathological", clamp_mismatch, [])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all postprocess tests passed")
sys.exit(1 if FAIL else 0)
