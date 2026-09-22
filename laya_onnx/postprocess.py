"""NumPy postprocessing for the ONNX path: logits -> laya's typed answer dicts.

`QTYPES`/`QTYPE_NAMES`, `confidence_from_probs`, `temp_bucket` and `clamp_temperature` are
copied verbatim from laya/common.py (QTYPES/QTYPE_NAMES: :11-12, confidence_from_probs: :210,
temp_bucket: :219, TEMP_MIN/TEMP_MAX: :228-229, clamp_temperature: :232). They live here rather
than in collate.py because collate() never needs the qtype-name mapping -- it passes `qtype`
through as a bare int for the graph -- while build_answers() needs the names to pick a
temperature bucket and to label a `score` question's legend.

`build_answers` lifts the per-question answer loop from laya/agent.py's torch path
(roughly :322-362 in the installed 0.3.5) and makes it framework-free:

- `self.temperature_by_options` / `self.temperature` become explicit parameters, since this
  module has no Agent instance to hold them.
- The `act` softmax that the torch path does once outside the loop, in torch
  (`torch.softmax(act.float(), -1)`, laya/agent.py:320), moves *inside* build_answers and is
  done with plain NumPy -- `act` arrives here as raw logits, not the already-softmaxed
  probabilities the torch loop reads from `act[r, 0]`.
- Unlike the torch path (which clamps temperatures once at Agent load time -- see
  laya/agent.py:206-210 -- and stores the clamped values on `self.temperature`/
  `self.temperature_by_options`), this function clamps the *looked-up* temperature itself, on
  every question, regardless of what the caller passed in. `laya-multilingual` ships no fitted
  temperatures at all, so a caller here will very often pass `temperature_by_options={}` and a
  bare `temperature` default straight from config with no separate "load-time clamp" step ever
  having run over it. Clamping at lookup time means an un-vetted temperature (the shipped
  `choice:11+` bucket of 0.1006, for instance) can never reach the softmax undamped no matter
  how the caller assembled its inputs.
"""
import math
from typing import Any, Dict, List

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published
# as 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest
# calibration needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Max-subtraction keeps exp() from overflowing on the -1e4 masked-out logits build_sequence's
    # marker masking fills unused option slots with (see laya/common.py:117); without it this
    # softmax is numerically identical in exact arithmetic but not in float32.
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def build_answers(
    question_ids: List[str],
    internal_questions: Dict[str, Dict[str, Any]],
    items: List[Dict[str, Any]],
    logits: np.ndarray,
    act: np.ndarray,
    temperature: List[float],
    temperature_by_options: Dict[str, float],
) -> Dict[str, Any]:
    """Turn one batch's raw model outputs into laya's per-question answer dicts.

    `question_ids[r]` / `items[r]` / `logits[r]` / `act[r]` all refer to the same row r of one
    batch -- the caller (Task 7's pipeline) is responsible for keeping them aligned, the same way
    `laya/agent.py`'s `ids` and `items` lists stay aligned by construction.
    """
    act_p = _softmax(act.astype(np.float32), axis=-1)

    answers: Dict[str, Any] = {}
    for r, qid in enumerate(question_ids):
        q = internal_questions[qid]
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = clamp_temperature(temperature_by_options.get(temp_bucket(qt, k), temperature[qt]))

        z = logits[r, :k] / t_scale
        p = _softmax(z, axis=-1)

        conf_score = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act_p[r, 0]), 4)}

        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf_score,
                "action": ext,
            }
        elif q["t"] == "score":
            exp_score = float((np.arange(k) * p).sum())
            answers[qid] = {
                "type": "score",
                "score": round(exp_score, 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf_score,
                "action": ext,
            }
        else:
            answers[qid] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                "action": ext,
            }

    return answers
