"""Offline tests for the int8/calibration machinery: ECE arithmetic, bucketing, the fit, the clamp.

What is and is not covered here, deliberately.

*Covered*: everything that is a pure function of numbers. The ECE a bucket reports for a
hand-computable set of probabilities. The invariance of accuracy under temperature (the property
that makes temperature scaling safe to apply at all, and the reason "accuracy didn't move" is
never evidence that calibration didn't). The recovery of a known temperature from labels sampled
at that temperature. The clamp firing on a fit that wants to sharpen. The fit/report split being
disjoint, per-bucket, deterministic, and refusing to fit a bucket it cannot hold data out of.
The two guards in the quantizer that fail before touching a weight.

*Not covered*: any number that depends on the checkpoint. Whether int8 costs 0.01 ECE or 0.10 on
`laya-multilingual` is a measurement, it needs 1.3 GB of weights and a Hub download, and it
belongs in a report, not in a test that pretends to be an assertion. `laya_onnx/bench/eval_ece.py`
has a `__main__` for that; this file never loads a model.

The tests below build rows in the shape `collect_logits` returns, which is how a synthetic case
and a real one stay interchangeable.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_onnx.bench.eval_ece import (                                       # noqa: E402
    bucket_metrics,
    metrics_from_rows,
    split_rows,
    temperature_for,
)
from laya_onnx.export.quantize_int8 import (                                 # noqa: E402
    _BODY_OP_TYPES,
    _EMBEDDING_INITIALIZER,
    _require_external_data,
    _strip_initializer_value_info,
    assert_embeddings_not_quantized,
    quantize,
    quantized_initializers,
)
from laya_onnx.export.refit_temps import (                                   # noqa: E402
    RAW_T_MIN,
    fit_temperature,
    fit_temperatures,
    write_temperatures,
)
from laya_onnx.postprocess import QTYPES                                     # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def row(qtype, k, logits, gold, bucket):
    return {"bucket": bucket, "qtype": qtype, "k": k,
            "logits": np.asarray(logits, dtype=np.float64), "gold": gold}


# --------------------------------------------------------------------- ECE arithmetic
# Ten noul rows whose logits give exactly p = [0.2, 0.8]: every row predicts option 1 with
# confidence 0.8, and exactly half of them are right. So accuracy is 0.5, mean confidence is
# 0.8, and every row lands in the same confidence bin -- which makes the ECE exactly the gap,
# |0.8 - 0.5| = 0.3, with no binning subtlety to hide an error in.
P8 = [0.0, math.log(4.0)]
noul_rows = ([row(QTYPES["noul"], 2, P8, 1, "noul:2")] * 5
             + [row(QTYPES["noul"], 2, P8, 0, "noul:2")] * 5)

m = bucket_metrics(noul_rows, [1.0, 1.0, 1.0], {})
check("ece/n", m["n"], 10)
check("ece/accuracy", m["accuracy"], 0.5)
check("ece/mean-confidence", m["mean_confidence"], 0.8)
check("ece/single-bin-gap", m["ece"], 0.3)
# For noul, laya publishes max(p, 1-p) as `confidence`, which here is the same 0.8, so the
# two estimators must agree exactly. If they ever diverge for k=2 one of them is wrong.
check("ece/published-confidence-matches-top-prob-for-noul",
      m["ece_published_confidence"], m["ece"])
check("ece/nll", m["nll"], round((-math.log(0.8) - math.log(0.2)) / 2, 4))

# A perfectly calibrated bucket must score 0: confidence 0.8 on rows that are right 80% of
# the time. (8 correct, 2 wrong.)
cal_rows = ([row(QTYPES["noul"], 2, P8, 1, "noul:2")] * 8
            + [row(QTYPES["noul"], 2, P8, 0, "noul:2")] * 2)
check("ece/perfect-calibration-is-zero", bucket_metrics(cal_rows, [1.0, 1.0, 1.0], {})["ece"], 0.0)

check("ece/empty-bucket", bucket_metrics([], [1.0, 1.0, 1.0], {}), {"n": 0})

# --------------------------------------------------------------------- temperature handling
# The property the whole approach rests on: a temperature cannot reorder the labels, so it
# cannot change accuracy. If this ever fails, a reported "the refit didn't cost accuracy" is
# meaningless, because accuracy was never at risk and something else moved it.
hot = bucket_metrics(noul_rows, [1.0, 1.0, 1.0], {"noul:2": 4.0})
check("temperature/accuracy-invariant", hot["accuracy"], m["accuracy"])
check_true("temperature/softens-confidence", hot["mean_confidence"] < m["mean_confidence"],
           "%r vs %r" % (hot["mean_confidence"], m["mean_confidence"]))
check("temperature/reported", hot["temperature"], 4.0)

# Resolution order, and the clamp, must match postprocess.build_answers exactly.
r11 = row(QTYPES["choice"], 20, [0.0] * 20, 0, "choice:11+")
check("temperature/bucket-overrides-default",
      temperature_for(r11, [1.0, 1.0, 1.0], {"choice:11+": 2.5}), 2.5)
check("temperature/falls-back-to-qtype-default",
      temperature_for(r11, [3.0, 1.0, 1.0], {}), 3.0)
# The shipped english-checkpoint value. It must never reach a softmax.
check("temperature/shipped-0.1006-is-clamped",
      temperature_for(r11, [1.0, 1.0, 1.0], {"choice:11+": 0.1006}), 0.5)
check("temperature/absurd-default-is-clamped",
      temperature_for(r11, [50.0, 1.0, 1.0], {}), 5.0)

# metrics_from_rows must split by bucket, not pool.
mixed = noul_rows + [row(QTYPES["choice"], 4, [2.0, 0.0, 0.0, 0.0], 0, "choice:3-5")] * 4
per = metrics_from_rows(mixed, [1.0, 1.0, 1.0])
check("buckets/keys", sorted(per), ["choice:3-5", "noul:2"])
check("buckets/counts", (per["noul:2"]["n"], per["choice:3-5"]["n"]), (10, 4))
check("buckets/k-recorded", per["choice:3-5"]["k"], 4)

# --------------------------------------------------------------------- the fit
# Sample labels from softmax(z / T_TRUE) and check the NLL fit recovers T_TRUE. This is the
# only honest way to test a fitter: a fit that merely "runs" tells you nothing about whether it
# found the minimum or a boundary.
T_TRUE = 2.0
rng = np.random.default_rng(20260921)
fit_rows = []
for _ in range(6000):
    z = rng.normal(0.0, 3.0, size=4)
    p = np.exp(z / T_TRUE - (z / T_TRUE).max())
    p = p / p.sum()
    fit_rows.append(row(QTYPES["choice"], 4, z, int(rng.choice(4, p=p)), "choice:3-5"))
t_hat = fit_temperature(fit_rows)
check_true("fit/recovers-known-temperature", abs(t_hat - T_TRUE) < 0.1,
           "fitted %.4f for a true %.2f" % (t_hat, T_TRUE))

# The fitted temperature must actually be a minimum of the objective it claims to minimise.
from laya_onnx.export.refit_temps import _nll                                # noqa: E402
check_true("fit/is-a-minimum-of-nll",
           _nll(fit_rows, t_hat) <= min(_nll(fit_rows, t_hat * 0.8),
                                        _nll(fit_rows, t_hat * 1.25)),
           "%.6f vs %.6f / %.6f" % (_nll(fit_rows, t_hat), _nll(fit_rows, t_hat * 0.8),
                                    _nll(fit_rows, t_hat * 1.25)))

# A bucket whose labels are always the (narrowly) top-scoring option wants an arbitrarily small
# temperature: sharpening the margin drives NLL toward zero with no minimum inside the range.
# This is exactly the shape of the shipped 0.1006 bucket, and the clamp is what stops it.
sharp = [row(QTYPES["choice"], 2, [0.0, 0.1], 1, "choice:2")] * 50
clamped, raw = fit_temperatures(sharp)
check_true("clamp/raw-fit-wants-to-sharpen", raw["choice:2"] < 0.5,
           "raw fit was %r" % raw["choice:2"])
check_true("clamp/raw-fit-hits-the-search-floor",
           abs(raw["choice:2"] - RAW_T_MIN) < 0.01, "raw fit was %r" % raw["choice:2"])
check("clamp/published-value-is-clamped", clamped["choice:2"], 0.5)

check("fit/multiple-buckets", sorted(fit_temperatures(fit_rows + sharp)[0]),
      ["choice:2", "choice:3-5"])

try:
    fit_temperature([])
    check("fit/empty-raises", "no raise", "ValueError")
except ValueError:
    PASS.append("fit/empty-raises")

# --------------------------------------------------------------------- the fit/report split
split_pool = ([row(QTYPES["noul"], 2, P8, i % 2, "noul:2") for i in range(40)]
              + [row(QTYPES["choice"], 4, [1.0, 0, 0, 0], 0, "choice:3-5") for _ in range(30)]
              + [row(QTYPES["choice"], 8, [1.0] + [0] * 7, 0, "choice:6-10") for _ in range(9)])
fit_half, report_half, unfittable = split_rows(split_pool, seed=0, min_per_bucket=20)

check("split/covers-everything", len(fit_half) + len(report_half), len(split_pool))
check("split/disjoint", len({id(r) for r in fit_half} & {id(r) for r in report_half}), 0)
check("split/too-small-bucket-is-unfittable", unfittable, ["choice:6-10"])
check("split/unfittable-rows-go-to-report-only",
      sum(1 for r in fit_half if r["bucket"] == "choice:6-10"), 0)
check("split/unfittable-rows-are-still-reported",
      sum(1 for r in report_half if r["bucket"] == "choice:6-10"), 9)
check("split/halves-per-bucket-noul",
      (sum(1 for r in fit_half if r["bucket"] == "noul:2"),
       sum(1 for r in report_half if r["bucket"] == "noul:2")), (20, 20))
check("split/halves-per-bucket-choice",
      (sum(1 for r in fit_half if r["bucket"] == "choice:3-5"),
       sum(1 for r in report_half if r["bucket"] == "choice:3-5")), (15, 15))

again = split_rows(split_pool, seed=0, min_per_bucket=20)
check("split/deterministic", [id(r) for r in again[0]], [id(r) for r in fit_half])
different = split_rows(split_pool, seed=1, min_per_bucket=20)
check_true("split/seed-changes-it", [id(r) for r in different[0]] != [id(r) for r in fit_half])

# --------------------------------------------------------------------- writing temperatures
with tempfile.TemporaryDirectory() as d:
    cfg = os.path.join(d, "rl_agent_config.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"max_len": 1024, "head_max_len": 256,
                   "temperature_by_options": {"noul:2": 1.5}}, f)
    write_temperatures(d, {"choice:3-5": 2.25})
    with open(cfg, encoding="utf-8") as f:
        after = json.load(f)
    # A merge, not a replacement: re-fitting one bucket must not delete the others.
    check("write/merges", after["temperature_by_options"],
          {"noul:2": 1.5, "choice:3-5": 2.25})
    check("write/keeps-other-keys", after["max_len"], 1024)

# --------------------------------------------------------------------- quantizer guards
with tempfile.TemporaryDirectory() as d:
    lonely = os.path.join(d, "model.onnx")
    with open(lonely, "wb") as f:
        f.write(b"not really a model")
    try:
        _require_external_data(lonely)
        check("quantize/missing-weights-raises", "no raise", "FileNotFoundError")
    except FileNotFoundError as e:
        # The message has to name the missing file, or the error is a scavenger hunt.
        check_true("quantize/missing-weights-raises", "model.onnx.data" in str(e),
                   "message was %r" % str(e))
    try:
        quantize(os.path.join(d, "nope.onnx"), os.path.join(d, "out.onnx"))
        check("quantize/missing-model-raises", "no raise", "FileNotFoundError")
    except FileNotFoundError:
        PASS.append("quantize/missing-model-raises")

# The value_info strip: exactly the entries that duplicate an initializer, and nothing else.
# This is the step that makes quantize_dynamic's shape-inference pass survive the real export
# (it raises "Inferred shape and existing shape differ in dimension 0: (772) vs (256)" without
# it), so a change that quietly stops stripping -- or starts stripping intermediates -- must
# fail here rather than 1.3 GB later.
import onnx                                                                  # noqa: E402
from onnx import TensorProto, helper                                         # noqa: E402

with tempfile.TemporaryDirectory() as d:
    w = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        initializer=[w],
        value_info=[helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
    )
    src = os.path.join(d, "m.onnx")
    onnx.save(helper.make_model(graph), src)
    staged = _strip_initializer_value_info(src)
    kept = sorted(vi.name for vi in onnx.load(staged).graph.value_info)
    check("strip/drops-initializer-value-info", kept, ["X"])
    check("strip/keeps-the-original", sorted(vi.name for vi in onnx.load(src).graph.value_info),
          ["W", "X"])
    check_true("strip/writes-beside-the-original",
               os.path.dirname(os.path.abspath(staged)) == os.path.dirname(os.path.abspath(src)),
               "staged at %r" % staged)
    check("strip/initializer-survives", [t.name for t in onnx.load(staged).graph.initializer],
          ["W"])

# --------------------------------------------------- the embedding-table exclusion
# The default `op_types_to_quantize` includes `Gather`, and this checkpoint's only large
# `Gather` operand is the 256k-row token-embedding table, which gets quantized per-tensor --
# one scale and one zero-point for 256,000 rows. `quantize_embeddings=False` restricts the op
# set to MatMul so that table stays fp32.
#
# It is worth being clear about what these tests do and do not certify. Sparing the table was
# measured and does **not** recover accuracy (`noul:2` lands on 0.7800 either way against
# fp32's 0.8833) -- see point 4 of quantize_int8's docstring. What is asserted below is only
# that the exclusion *happens*, verified from the written graph rather than from the flag that
# was passed, because that is what made the comparison trustworthy: a flag that silently did
# nothing would have produced two identical graphs and a confident null result.
check("exclusion/only-matmul-is-quantized", list(_BODY_OP_TYPES), ["MatMul"])


def _emb_graph(embedding_tensor, extra=()):
    """A one-node graph carrying `embedding_tensor` (or nothing) as its embedding initializer."""
    inits = ([embedding_tensor] if embedding_tensor is not None else []) + list(extra)
    g = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        initializer=inits)
    return helper.make_model(g)


with tempfile.TemporaryDirectory() as d:
    ok_path = os.path.join(d, "ok.onnx")
    body_weight = helper.make_tensor("val_1_quantized", TensorProto.INT8, [2, 2],
                                     bytes([1, 2, 3, 4]), raw=True)
    onnx.save(_emb_graph(
        helper.make_tensor(_EMBEDDING_INITIALIZER, TensorProto.FLOAT, [2, 2],
                           [1.0, 2.0, 3.0, 4.0]),
        extra=[body_weight]), ok_path)
    try:
        assert_embeddings_not_quantized(ok_path)
        PASS.append("exclusion/fp32-embedding-passes")
    except RuntimeError as e:
        FAIL.append("exclusion/fp32-embedding-passes: raised %r" % (e,))
    # The body weight must still register as quantized -- an "exclusion" that excluded
    # everything would also pass the check above, and would be a 1290 MB "int8" model.
    check("exclusion/body-weight-still-reported-quantized",
          [(n, dt) for n, dt, _ in quantized_initializers(ok_path)],
          [("val_1_quantized", "INT8")])

    bad_path = os.path.join(d, "bad.onnx")
    onnx.save(_emb_graph(helper.make_tensor(
        _EMBEDDING_INITIALIZER + "_quantized", TensorProto.UINT8, [2, 2],
        bytes([1, 2, 3, 4]), raw=True)), bad_path)
    try:
        assert_embeddings_not_quantized(bad_path)
        FAIL.append("exclusion/quantized-embedding-raises: no raise")
    except RuntimeError as e:
        check_true("exclusion/quantized-embedding-raises", "token-embedding" in str(e),
                   "message was %r" % str(e))

    # A graph this check does not recognise must fail loudly rather than vacuously pass: a
    # silent pass would be a guard that certifies every unknown model as safe.
    unknown = os.path.join(d, "unknown.onnx")
    onnx.save(_emb_graph(None), unknown)
    try:
        assert_embeddings_not_quantized(unknown)
        FAIL.append("exclusion/unknown-graph-raises: no raise")
    except RuntimeError as e:
        check_true("exclusion/unknown-graph-raises", _EMBEDDING_INITIALIZER in str(e),
                   "message was %r" % str(e))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all calibration tests passed")
sys.exit(1 if FAIL else 0)
