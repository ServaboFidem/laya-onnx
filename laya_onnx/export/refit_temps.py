"""Fit one temperature per `temp_bucket` against a quantized graph, and write them into its config.

**Why refit at all, and why against int8.** `laya-multilingual` ships `temperature_by_options`
empty and `temperature = [1.0, 1.0, 1.0]`. There is no calibration here to preserve -- the
`confidence` field the runtime publishes for this checkpoint is, out of the box, an unfitted
softmax. So this is not damage control after quantization; it is the first fit this checkpoint
has ever had. And since int8 weights move the logits, the fit has to be done on the graph that
actually ships: a temperature fitted on fp32 logits and applied to int8 logits is a correction
for the wrong error.

**Temperature scaling, specifically.** One scalar per bucket, dividing the logits before the
softmax. It is the weakest calibration map there is -- it cannot reorder the labels, so accuracy
is invariant under it by construction, and it has exactly one parameter per bucket to overfit
with. That weakness is the point: a richer map (vector scaling, isotonic regression) would fit
the held-out half better and would also be able to hide a genuinely mis-ranked model behind a
flattering reliability curve.

**The objective is NLL, not ECE.** ECE is a binned statistic: it is piecewise-constant in the
temperature, has zero gradient almost everywhere, and can be driven to a small value by a
temperature that is simply bad (push every confidence into one bin and the bin's average error
is all that remains). Negative log-likelihood is smooth, strictly proper, and convex in the
inverse temperature, so a one-dimensional search on it has a single answer. ECE is what gets
*reported* -- by `laya_onnx/bench/eval_ece.py`, on the half of the data this module never saw.

**The clamp is not optional.** `postprocess.clamp_temperature` confines the result to
[0.5, 5.0]. The cautionary case is shipped in the `english` checkpoint: a `choice:11+` bucket of
0.1006, which multiplies the logits by ~10 and republishes a 0.24 top probability as 0.99. An
unconstrained NLL fit on a small or skewed bucket can land there, and a caller gating on
`confidence` would be told a coin flip is a certainty. Both the raw and the clamped value are
returned so a report can say plainly that a bucket hit the rail -- a clamped bucket is a bucket
whose calibration was *not* achieved, and it should be described that way rather than quietly
included in an average.
"""
import argparse
import json
import math
import os
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..bench.eval_ece import collect_logits, split_rows
from ..postprocess import clamp_temperature

# The search range for the *raw* fit, deliberately far wider than the [0.5, 5.0] the result is
# clamped to. Fitting inside the clamp would make "hit the rail" unobservable: every bucket
# would report a plausible in-range temperature and nothing would distinguish a genuine 0.51
# from a fit that wanted 0.08. Searching wide and clamping after keeps that distinction, which
# is the difference between a calibrated bucket and one that merely looks calibrated.
RAW_T_MIN = 0.05
RAW_T_MAX = 20.0

_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _nll(rows: Sequence[Dict[str, Any]], t: float) -> float:
    """Mean negative log-likelihood of the gold labels at temperature `t`."""
    total = 0.0
    for row in rows:
        z = row["logits"] / t
        z = z - z.max()
        total += float(np.log(np.exp(z).sum()) - z[row["gold"]])
    return total / len(rows)


def fit_temperature(rows: Sequence[Dict[str, Any]],
                    lo: float = RAW_T_MIN, hi: float = RAW_T_MAX) -> float:
    """The raw (unclamped) temperature minimising NLL over `rows`.

    Golden-section search on the inverse temperature u = 1/t, over [1/hi, 1/lo]. The search runs
    on u rather than t because the softmax cross-entropy is convex in u -- golden section needs
    unimodality, and convexity in t is not guaranteed. 100 iterations shrink the bracket by
    0.618**100, which is far past float64 resolution; the loop is cheap because `rows` holds
    logits, not a model.
    """
    if not rows:
        raise ValueError("cannot fit a temperature on zero rows")

    a, b = 1.0 / hi, 1.0 / lo
    c, d = b - _PHI * (b - a), a + _PHI * (b - a)
    fc, fd = _nll(rows, 1.0 / c), _nll(rows, 1.0 / d)
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _PHI * (b - a)
            fc = _nll(rows, 1.0 / c)
        else:
            a, c, fc = c, d, fd
            d = a + _PHI * (b - a)
            fd = _nll(rows, 1.0 / d)
    return 1.0 / ((a + b) / 2.0)


def fit_temperatures(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Fit every bucket present in `rows`. Returns (clamped mapping, raw mapping).

    `rows` must already be the *fit* half of a split (see `eval_ece.split_rows`); this function
    does not split, because a function that both splits and fits can too easily be called in a
    way that fits on everything.
    """
    clamped: Dict[str, float] = {}
    raw: Dict[str, float] = {}
    for bucket in sorted({r["bucket"] for r in rows}):
        group = [r for r in rows if r["bucket"] == bucket]
        t_raw = fit_temperature(group)
        raw[bucket] = round(float(t_raw), 4)
        clamped[bucket] = round(clamp_temperature(t_raw), 4)
    return clamped, raw


def refit(agent, dataset, seed: int = 0, min_per_bucket: int = 20) -> Dict[str, float]:
    """Fit `temperature_by_options` for `agent` on half of `dataset`.

    `agent` must be the *quantized* agent when the target is an int8 export -- see the module
    docstring. Returns the clamped mapping, ready to be written into an export's
    `rl_agent_config.json` (`write_temperatures`) after being scored on the other half
    (`eval_ece.metrics_from_rows` over the report rows).

    Buckets with fewer than `min_per_bucket` examples are absent from the returned mapping
    rather than fitted on what little they have; the runtime falls back to the per-qtype
    default for any bucket it does not find, so an omitted bucket is an *unfitted* bucket, which
    is the honest state to leave it in.
    """
    rows = collect_logits(agent, dataset)
    fit_rows, _report_rows, _unfittable = split_rows(rows, seed=seed,
                                                     min_per_bucket=min_per_bucket)
    clamped, _raw = fit_temperatures(fit_rows)
    return clamped


def write_temperatures(model_dir: str, temperature_by_options: Dict[str, float]) -> str:
    """Merge `temperature_by_options` into `model_dir`'s rl_agent_config.json. Returns the path.

    A merge, not a replacement: a later study that only re-fits the noul buckets must not silently
    delete the choice ones. The keys are the `temp_bucket` strings, which is exactly what
    `postprocess.build_answers` looks up.
    """
    path = os.path.join(model_dir, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    existing = dict(cfg.get("temperature_by_options") or {})
    existing.update({k: float(v) for k, v in temperature_by_options.items()})
    cfg["temperature_by_options"] = existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def main(argv=None) -> int:
    """Write a previously measured mapping into an export's config.

    The measurement itself is `python -m laya_onnx.bench.eval_ece`, which owns the dataset
    fetch and the fit/report split; this entry point exists so that writing the result into a
    checkpoint is a separate, deliberate act rather than a side effect of measuring.
    """
    ap = argparse.ArgumentParser(description="write fitted temperatures into an export config")
    ap.add_argument("model_dir")
    ap.add_argument("temperatures", help="JSON object, or a path to one, mapping bucket -> float")
    a = ap.parse_args(argv)
    raw = a.temperatures
    if os.path.exists(raw):
        with open(raw, encoding="utf-8") as f:
            raw = f.read()
    mapping = json.loads(raw)
    print("wrote %s" % write_temperatures(a.model_dir, mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
