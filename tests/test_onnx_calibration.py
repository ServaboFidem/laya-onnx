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

# transformers probes for TensorFlow at import; with TF present its abseil runtime can deadlock
# model construction. This suite reaches torch and transformers transitively through
# laya.common (imported by laya_onnx.bench.eval_ece), so it needs the same guard every other
# torch-touching entry point in this repo sets. CI sets them job-wide; a local `python
# tests/test_onnx_calibration.py` does not.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_onnx.bench.eval_ece import (                                       # noqa: E402
    accuracy_of,
    bootstrap_ci,
    bucket_metrics,
    collect_logits,
    ece_of,
    measure_ece,
    metrics_from_rows,
    paired_mcnemar,
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
    REACHABLE_BUCKETS,
    fit_temperature,
    fit_temperatures,
    rail_status,
    refit,
    refit_report,
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


# ------------------------------------------------- the brief's own interfaces, on a fake agent
# Everything above builds its rows by hand, which tests the arithmetic but leaves the three
# functions the plan actually specifies -- `collect_logits`, `measure_ece`, `refit` -- with no
# coverage at all, and never checks that `collect_logits` emits the shape the rest of the module
# assumes. Worse, the two guards that protect the *validity* of every published number (option
# truncation, gold range) were unexercised: a silently truncated option list drops the gold
# label for some examples and turns an accuracy into a measurement of the truncation.
#
# A fake tokenizer and a fake session are enough, and the repo has precedent for exactly this
# (tests/test_shortlist.py fakes `embed_fn` and mocks `predict`). No weights, no Hub, no ORT.


class FakeTok:
    """Whitespace tokenizer with the five attributes `build_sequence` reads.

    Token ids come from a character sum rather than `hash()`, which is salted per process and
    would make this test's sequence lengths differ between runs.
    """
    mask_token = "[MASK]"
    mask_token_id = 4
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (sum(map(ord, w)) % 50) for w in text.split()]}


class FakeSession:
    """Returns the next row of `table` per batch row, padded to the batch's marker width."""

    def __init__(self, table):
        self.table = list(table)
        self.calls = 0
        self.batches = 0

    def run(self, batch):
        n, kmax = batch["marker_pos"].shape
        self.batches += 1
        logits = np.full((n, kmax), -1e4, dtype=np.float32)
        act = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            vals = self.table[self.calls]
            self.calls += 1
            logits[i, :len(vals)] = vals
        return logits, act


class FakeAgent:
    def __init__(self, table, max_len=256, head_max_len=128):
        self.tok = FakeTok()
        self.session = FakeSession(table)
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.temperature = [1.0, 1.0, 1.0]
        self.temperature_by_options = {}


def choice_q(keys):
    return {"type": "choice", "instructions": "which one", "criteria": {k: None for k in keys}}


def noul_q():
    return {"type": "noul", "instructions": "is it so"}


# --- collect_logits emits the documented row shape ---------------------------------------
ag = FakeAgent([[3.0, 1.0, 0.0], [0.5, 2.5]])
ds = [{"state": {"s": "alpha beta"}, "questions": {"q": choice_q(["a", "b", "c"])},
       "gold": {"q": 0}},
      {"state": "gamma delta", "questions": {"q": noul_q()}, "gold": {"q": 1}}]
got = collect_logits(ag, ds)
check("collect/one-row-per-question", len(got), 2)
check("collect/bucket", [r["bucket"] for r in got], ["choice:3-5", "noul:2"])
check("collect/k", [r["k"] for r in got], [3, 2])
check("collect/qtype", [r["qtype"] for r in got], [QTYPES["choice"], QTYPES["noul"]])
check("collect/gold", [r["gold"] for r in got], [0, 1])
# float64 matters: the NLL fit runs on these and float32 accumulation noise is the same order
# as the int8-vs-fp32 differences the study is measuring.
check("collect/logits-are-float64", [r["logits"].dtype for r in got],
      [np.dtype("float64")] * 2)
# Sliced to k, so the -1e4 padding the graph emits for unused marker slots never reaches a
# softmax or an NLL.
check("collect/logits-sliced-to-k", [list(r["logits"]) for r in got],
      [[3.0, 1.0, 0.0], [0.5, 2.5]])
check("collect/one-batch-per-record", ag.session.batches, 2)

# --- the guards that protect every published number ---------------------------------------
# A question whose markers do not all survive `max_len` has a truncated option list. Measuring
# it would drop the gold label for some examples and not others.
narrow = FakeAgent([[0.0] * 6], max_len=14, head_max_len=200)
try:
    collect_logits(narrow, [{"state": "x", "questions": {"q": choice_q(list("abcdef"))},
                             "gold": {"q": 0}}])
    FAIL.append("collect/truncated-options-raise: no raise")
except ValueError as e:
    check_true("collect/truncated-options-raise", "head_max_len" in str(e),
               "message was %r" % str(e))

# The exported graph was traced through laya/common.py's topk(2) branch.
try:
    collect_logits(FakeAgent([[0.0]]), [{"state": "x", "questions": {"q": choice_q(["only"])},
                                         "gold": {"q": 0}}])
    FAIL.append("collect/single-option-raises: no raise")
except ValueError as e:
    check_true("collect/single-option-raises", "at least 2" in str(e), "message was %r" % str(e))

# A gold index outside the option set would score as permanently wrong and depress accuracy
# silently.
try:
    collect_logits(FakeAgent([[1.0, 2.0, 3.0]]),
                   [{"state": "x", "questions": {"q": choice_q(["a", "b", "c"])},
                     "gold": {"q": 7}}])
    FAIL.append("collect/gold-out-of-range-raises: no raise")
except ValueError as e:
    check_true("collect/gold-out-of-range-raises", "outside" in str(e), "message was %r" % str(e))

# --- measure_ece end to end ----------------------------------------------------------------
# Two noul rows at p = [0.2, 0.8] (logits differ by log 4), one right and one wrong: accuracy
# 0.5, confidence 0.8, one bin, so ECE is exactly 0.3 -- the same hand-computable case the
# synthetic rows use above, now driven through the real pipeline.
pair = [0.0, math.log(4.0)]
ece_ds = [{"state": "x", "questions": {"q": noul_q()}, "gold": {"q": 1}},
          {"state": "y", "questions": {"q": noul_q()}, "gold": {"q": 0}}]
res = measure_ece(FakeAgent([pair, pair]), ece_ds)
check("measure_ece/buckets", sorted(res), ["noul:2"])
check("measure_ece/n", res["noul:2"]["n"], 2)
check("measure_ece/accuracy", res["noul:2"]["accuracy"], 0.5)
check("measure_ece/ece", res["noul:2"]["ece"], 0.3)
check("measure_ece/keys", sorted(res["noul:2"]),
      ["accuracy", "ece", "ece_published_confidence", "k", "mean_confidence",
       "mean_published_confidence", "n", "nll", "temperature"])
# An override must reach the metric, or "after fitting" numbers would silently be "before".
hot_res = measure_ece(FakeAgent([pair, pair]), ece_ds, temperature_by_options={"noul:2": 4.0})
check("measure_ece/override-applies", hot_res["noul:2"]["temperature"], 4.0)
check_true("measure_ece/override-softens",
           hot_res["noul:2"]["mean_confidence"] < res["noul:2"]["mean_confidence"])

# --- refit / refit_report -------------------------------------------------------------------
# 40 noul records, enough to clear min_per_bucket on both halves.
refit_ds = [{"state": "s%d" % i, "questions": {"q": noul_q()}, "gold": {"q": i % 2}}
            for i in range(40)]
mapping = refit(FakeAgent([pair] * 40), refit_ds)
check("refit/returns-bucket-mapping", sorted(mapping), ["noul:2"])
check_true("refit/temperature-is-clamped",
           0.5 <= mapping["noul:2"] <= 5.0, "got %r" % mapping["noul:2"])
rep = refit_report(FakeAgent([pair] * 40), refit_ds)
check("refit/report-agrees-with-refit", rep["temperature_by_options"], mapping)
check("refit/report-exposes-raw", sorted(rep["raw"]), ["noul:2"])
check("refit/report-exposes-rail", sorted(rep["rail"]), ["noul:2"])
check("refit/halves-are-equal", (rep["n_fit"], rep["n_report"]), (20, 20))
# A bucket too small to hold data out must be reported, not quietly fitted on everything.
small = refit_report(FakeAgent([pair] * 6),
                     [{"state": "s", "questions": {"q": noul_q()}, "gold": {"q": 0}}] * 6)
check("refit/too-small-bucket-unfitted", small["unfitted_buckets"], ["noul:2"])
check("refit/too-small-bucket-absent-from-mapping", small["temperature_by_options"], {})
check("refit/too-small-bucket-still-seen", small["buckets_seen"], ["noul:2"])

# --- rail status ------------------------------------------------------------------------
# The measured case: int8-body's noul:2 fitted at 4.9490 against a 5.0 bound. It was never
# clamped, so a clamp-only check calls it healthy; it is not a converged optimum.
check("rail/measured-4.949-is-near-rail", rail_status(4.9490), "near_rail")
check("rail/comfortably-inside", rail_status(2.0), "ok")
check("rail/just-above-upper-bound", rail_status(6.0), "clamped")
check("rail/the-shipped-0.1006", rail_status(0.1006), "clamped")
check("rail/just-inside-lower-bound", rail_status(0.55), "near_rail")

# --- reachable buckets / provenance -------------------------------------------------------
# Nine, not twelve: render_options builds a noul question's options as a fixed [false, true]
# pair, so noul can only ever be k=2.
check("buckets/reachable-count", len(REACHABLE_BUCKETS), 9)
check("buckets/noul-only-has-k2", [b for b in REACHABLE_BUCKETS if b.startswith("noul")],
      ["noul:2"])

with tempfile.TemporaryDirectory() as d:
    cfg = os.path.join(d, "rl_agent_config.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"max_len": 1024, "head_max_len": 256, "temperature_by_options": {}}, f)
    write_temperatures(d, {"choice:3-5": 1.4634, "noul:2": 3.5028},
                       provenance={"graph": "fp32", "datasets": ["ag_news"]})
    with open(cfg, encoding="utf-8") as f:
        after = json.load(f)
    prov = after["temperature_by_options_provenance"]
    check("provenance/caller-fields-kept", prov["graph"], "fp32")
    check("provenance/fitted", prov["fitted_buckets"], ["choice:3-5", "noul:2"])
    # The half that matters: a config with two entries looks calibrated until it says which
    # seven buckets publish a raw softmax.
    check("provenance/unfitted-is-the-complement", prov["unfitted_buckets"],
          [b for b in REACHABLE_BUCKETS if b not in ("choice:3-5", "noul:2")])
    check_true("provenance/explains-the-fallback", "raw" in prov["unfitted_behaviour"])
    check_true("provenance/timestamped", prov["written_utc"].startswith("20"))
    # The runtime contract: OnnxAgent reads only temperature_by_options, so the extra key is
    # additive and an older runtime ignores it.
    check("provenance/mapping-unaffected", after["temperature_by_options"],
          {"choice:3-5": 1.4634, "noul:2": 3.5028})

# --- McNemar --------------------------------------------------------------------------
# Paired, because both graphs answered the same questions; an unpaired proportion test would
# overstate the variance badly.
# Same question (same gold) answered by two graphs, so the pair differs in its *logits*, not
# in its label -- which is exactly the situation the test is for.
right = row(QTYPES["noul"], 2, [0.0, 1.0], 1, "noul:2")     # predicts 1, gold 1 -> correct
wrong = row(QTYPES["noul"], 2, [1.0, 0.0], 1, "noul:2")     # predicts 0, gold 1 -> incorrect
m10 = paired_mcnemar([right] * 10, [wrong] * 10)
check("mcnemar/discordant-count", (m10["b_only_a_right"], m10["c_only_b_right"]), (10, 0))
# Exact two-sided binomial: 2 * C(10,0) / 2^10.
check("mcnemar/exact-p", round(m10["p_value"], 10), round(2.0 / 1024, 10))
check("mcnemar/no-discordance-is-p1", paired_mcnemar([right] * 5, [right] * 5)["p_value"], 1.0)
check("mcnemar/reports-both-accuracies",
      (m10["accuracy_a"], m10["accuracy_b"]), (1.0, 0.0))
try:
    paired_mcnemar([right] * 3, [right] * 4)
    FAIL.append("mcnemar/length-mismatch-raises: no raise")
except ValueError:
    PASS.append("mcnemar/length-mismatch-raises")
# Misaligned rows would silently compare different questions to each other.
# A differing gold or bucket at the same position means the two lists are not the same
# questions in the same order, which would silently compare unrelated answers.
for label, other in (("bucket", row(QTYPES["noul"], 2, [0.0, 1.0], 1, "choice:2")),
                     ("gold", row(QTYPES["noul"], 2, [0.0, 1.0], 0, "noul:2"))):
    try:
        paired_mcnemar([right], [other])
        FAIL.append("mcnemar/misaligned-%s-raises: no raise" % label)
    except ValueError as e:
        check_true("mcnemar/misaligned-%s-raises" % label, "not aligned" in str(e),
                   "message was %r" % str(e))

# --- bootstrap ---------------------------------------------------------------------------
mixed_rows = [right] * 8 + [wrong] * 2
lo, hi = bootstrap_ci(mixed_rows, accuracy_of, n_boot=500, seed=3)
check_true("bootstrap/brackets-the-estimate", lo <= accuracy_of(mixed_rows) <= hi,
           "%.3f not in [%.3f, %.3f]" % (accuracy_of(mixed_rows), lo, hi))
check("bootstrap/deterministic", bootstrap_ci(mixed_rows, accuracy_of, n_boot=500, seed=3),
      (lo, hi))
check_true("bootstrap/seed-changes-it",
           bootstrap_ci(mixed_rows, accuracy_of, n_boot=500, seed=4) != (lo, hi))
# A degenerate sample has no sampling variation, so the interval must collapse.
check("bootstrap/no-variance-no-width",
      bootstrap_ci([right] * 10, accuracy_of, n_boot=200, seed=5), (1.0, 1.0))
check("bootstrap/accuracy-is-temperature-free", accuracy_of(mixed_rows), 0.8)
check("bootstrap/ece-matches-bucket-metrics", round(ece_of(noul_rows, 1.0), 4),
      bucket_metrics(noul_rows, [1.0, 1.0, 1.0], {})["ece"])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all calibration tests passed")
sys.exit(1 if FAIL else 0)
