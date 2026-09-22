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
import datetime
import json
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..postprocess import TEMP_MAX, TEMP_MIN, clamp_temperature

# `collect_logits` / `split_rows` are imported inside `refit_report`, not here. `..bench.eval_ece`
# imports `laya.common` for the upstream ECE estimator, which imports torch at module scope --
# ~2 GB of dependency for a module whose CLI (`python -m laya_onnx.export.refit_temps`) does
# nothing but write a JSON key into an existing config. That CLI is the one thing under export/
# a *serving* operator plausibly runs, so `write_temperatures` and `main()` stay reachable
# without torch installed at all; only `refit`/`refit_report`, which need a live agent and a
# labelled dataset anyway, pull it in. This file is under export/ and is permitted to import
# torch -- it just should not do so for free.

# The search range for the *raw* fit, deliberately far wider than the [0.5, 5.0] the result is
# clamped to. Fitting inside the clamp would make "hit the rail" unobservable: every bucket
# would report a plausible in-range temperature and nothing would distinguish a genuine 0.51
# from a fit that wanted 0.08. Searching wide and clamping after keeps that distinction, which
# is the difference between a calibrated bucket and one that merely looks calibrated.
RAW_T_MIN = 0.05
RAW_T_MAX = 20.0

# How close to a clamp bound counts as "at the rail". A fit that lands within 2% of [0.5, 5.0]
# is reporting that the minimum it wanted lies outside the range it was allowed -- or so close
# to the edge that a slightly different sample would have clamped it. Either way the number is
# a boundary artifact, not a converged optimum, and a caller that cannot tell the difference
# will average it in with the honest ones. Measured instance: this checkpoint's int8-body
# `noul:2` bucket fits at 4.9490, 99% of the way to the upper bound without triggering the
# clamp at all.
RAIL_TOLERANCE = 0.02

# Every bucket string `postprocess.temp_bucket` can actually produce, so `write_temperatures`
# can say which ones a fitted mapping does *not* cover. Nine, not twelve: `choice` and `score`
# each reach all four size classes, but `sequence.render_options` builds a `noul` question's
# options as a fixed `[false, true]` pair (laya_onnx/sequence.py:41-44), so k is always 2 there
# and `noul:3-5` / `noul:6-10` / `noul:11+` are unreachable by construction.
_SIZE_CLASSES = ["2", "3-5", "6-10", "11+"]
REACHABLE_BUCKETS = tuple(
    ["choice:%s" % z for z in _SIZE_CLASSES]
    + ["score:%s" % z for z in _SIZE_CLASSES]
    + ["noul:2"]
)

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


def rail_status(raw: float, lo: float = TEMP_MIN, hi: float = TEMP_MAX,
                tolerance: float = RAIL_TOLERANCE) -> str:
    """`"ok"`, `"clamped"`, or `"near_rail"` for one raw fitted temperature.

    `near_rail` exists because `clamped` alone under-reports the problem. A fit that wanted
    4.9490 against an upper bound of 5.0 was never clamped, so a clamp-only check calls it
    healthy -- but it is the same boundary artifact as one that wanted 6.0, arrived at by a
    sample that happened to fall just inside. Both mean the optimum is at or past the edge of
    the range the caller is willing to publish.
    """
    if raw < lo or raw > hi:
        return "clamped"
    span = hi - lo
    if raw - lo <= tolerance * span or hi - raw <= tolerance * span:
        return "near_rail"
    return "ok"


def refit_report(agent, dataset, seed: int = 0,
                 min_per_bucket: int = 20) -> Dict[str, Any]:
    """`refit` with its working shown: the mapping plus everything `refit` has to throw away.

    Returns `{"temperature_by_options", "raw", "rail", "unfitted_buckets", "n_fit", "n_report",
    "buckets_seen"}`.

    `refit` returns a bare `dict[str, float]` because that is the interface the plan specifies
    and the shape `write_temperatures` consumes. But a bare mapping cannot say that `noul:2`
    fitted at 4.9490 against a 5.0 bound, or that `score:3-5` was dropped for having too few
    rows -- and a caller who cannot see either will write a mapping into a checkpoint believing
    every entry converged. So the diagnostics are not deleted, they are moved here, and `refit`
    is a thin wrapper over this.
    """
    # Deferred: see the note beside this module's imports. Reaching a live agent and a labelled
    # dataset already implies the measurement environment; `main()`'s JSON write does not.
    from ..bench.eval_ece import collect_logits, split_rows

    rows = collect_logits(agent, dataset)
    fit_rows, report_rows, unfittable = split_rows(rows, seed=seed,
                                                   min_per_bucket=min_per_bucket)
    clamped, raw = fit_temperatures(fit_rows)
    return {
        "temperature_by_options": clamped,
        "raw": raw,
        "rail": {b: rail_status(raw[b]) for b in raw},
        "unfitted_buckets": unfittable,
        "n_fit": len(fit_rows),
        "n_report": len(report_rows),
        "buckets_seen": sorted({r["bucket"] for r in rows}),
    }


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

    This return type is lossy by design -- it is the mapping and nothing else. Use
    `refit_report` for the raw fits, the rail status of each, and which buckets were dropped.
    """
    return refit_report(agent, dataset, seed=seed,
                        min_per_bucket=min_per_bucket)["temperature_by_options"]


def write_temperatures(model_dir: str, temperature_by_options: Dict[str, float],
                       provenance: Optional[Dict[str, Any]] = None) -> str:
    """Merge `temperature_by_options` into `model_dir`'s rl_agent_config.json. Returns the path.

    A merge, not a replacement: a later study that only re-fits the noul buckets must not
    silently delete the choice ones. The keys are the `temp_bucket` strings, which is exactly
    what `postprocess.build_answers` looks up.

    **A partial mapping is written alongside a record of what it does not cover.** A config
    carrying four fitted temperatures looks calibrated. It is not: `build_answers` falls back to
    the per-qtype default for any bucket it cannot find, so the five unwritten buckets publish a
    raw, unfitted softmax -- and nothing in a bare `{"choice:3-5": 1.46, ...}` tells the next
    reader which is which, on what data, or against which graph. So this function also writes
    `temperature_by_options_provenance`, merging in whatever the caller passes (the graph, the
    datasets, the split) and adding the facts it can derive itself: which reachable buckets are
    covered and which are not.

    The provenance key is additional, not structural -- `OnnxAgent` reads only
    `temperature_by_options`, so an older runtime ignores it and nothing breaks.
    """
    path = os.path.join(model_dir, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    existing = dict(cfg.get("temperature_by_options") or {})
    existing.update({k: float(v) for k, v in temperature_by_options.items()})
    cfg["temperature_by_options"] = existing

    record = dict(cfg.get("temperature_by_options_provenance") or {})
    record.update(provenance or {})
    record["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    record["fitted_buckets"] = sorted(existing)
    # The important half: what a reader would otherwise have to work out for themselves.
    record["unfitted_buckets"] = [b for b in REACHABLE_BUCKETS if b not in existing]
    record["unfitted_behaviour"] = (
        "temp_bucket strings absent from temperature_by_options fall back to the per-qtype "
        "default in `temperature` (1.0 for this checkpoint), i.e. an unfitted raw softmax.")
    cfg["temperature_by_options_provenance"] = record

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
    ap.add_argument("--provenance", default=None,
                    help="JSON object (or path to one) recording where these came from: the "
                         "graph, the datasets, the split. Written alongside the mapping so a "
                         "partial calibration cannot later be mistaken for a complete one.")
    a = ap.parse_args(argv)

    def _load(v):
        if v is None:
            return None
        if os.path.exists(v):
            with open(v, encoding="utf-8") as f:
                v = f.read()
        return json.loads(v)

    mapping = _load(a.temperatures)
    path = write_temperatures(a.model_dir, mapping, provenance=_load(a.provenance))
    with open(path, encoding="utf-8") as f:
        record = json.load(f)["temperature_by_options_provenance"]
    print("wrote %s" % path)
    print("  fitted:   %s" % ", ".join(record["fitted_buckets"]))
    print("  UNFITTED: %s  (these publish a raw, unfitted softmax)"
          % (", ".join(record["unfitted_buckets"]) or "none"))
    if a.provenance is None:
        print("  no --provenance given; the config records only the bucket coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
