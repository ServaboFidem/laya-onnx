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

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all postprocess tests passed")
sys.exit(1 if FAIL else 0)
