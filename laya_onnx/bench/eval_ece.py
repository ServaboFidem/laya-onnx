"""Per-bucket calibration measurement for an ONNX laya agent.

**Why this module is the point of the int8 task.** Quantizing an encoder to int8 perturbs the
pre-softmax logits. Argmax is robust to that -- a label that won by 2 nats still wins by 1.9 --
so accuracy barely moves and every structural test still passes. The *softmax of* those logits
is not robust to it, and the softmax is what laya publishes. A model that names the same label
while reporting 0.95 where it should report 0.70 has failed at the only thing this library is
for. So "the answers look right" is not evidence about quantization; a measured ECE against a
measured fp32 baseline is.

**Buckets, not one global number.** laya scales temperature per `(question type, option count)`
bucket (`postprocess.temp_bucket`), because a 2-way noul and a 77-way choice are miscalibrated
in different directions and by different amounts. A single pooled ECE averages those into a
number that describes no bucket, and it is the pooled number that most easily flatters a bad
model. Everything here is reported per bucket, with its own `n`.

**Two confidences are reported, deliberately.** `ece` uses the top-1 probability, which is the
standard ECE estimator and the one the numbers in this repository's README and
`research/results/*.json` were computed with -- comparability is the whole reason
`laya.common.ece_score` is imported here rather than rebinned locally. But the number a caller
actually gates on is the `confidence` field laya publishes, which for `choice` and `score` is a
normalized-entropy confidence, not the top probability. `ece_published_confidence` measures that
field against the same correctness. For `noul` (k=2) the two coincide by construction, since
laya publishes `max(p, 1-p)` there.

**No data fetching.** `measure_ece(agent, dataset)` takes its examples as an argument (see this
package's `__init__`). The `__main__` block at the bottom is where the Hub download lives.

Dataset format -- an iterable of records::

    {"state": <str | dict>, "questions": {qid: <laya question dict>}, "gold": {qid: <int>}}

`gold` is an integer index into that question's option list: for `choice` the index of the
correct key in `criteria` order, for `score` the correct level, for `noul` 1 for true and 0 for
false (matching `p[1]` being P(true) in `postprocess.build_answers`).
"""
import json
import math
import os
import random
import sys

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# transformers probes for TensorFlow at import and its abseil runtime can deadlock model
# construction; laya is torch-only and every entry point in this repository sets these. This
# module reaches torch transitively through `laya.common`, so it is one of them -- CI sets them
# job-wide, but the manual local run that produces the published numbers is otherwise
# unprotected, and that run is where the numbers come from.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

# laya.common drags in torch. That is allowed here and nowhere else under laya_onnx/ (see this
# package's __init__): bench/ is measurement, not the serving path. Reusing the upstream
# estimator rather than rebinning locally is what makes these ECE numbers comparable to the
# ones already published for the torch path.
from laya.common import ece_score            # noqa: E402

from ..collate import collate                # noqa: E402
from ..postprocess import (                  # noqa: E402
    QTYPES,
    clamp_temperature,
    confidence_from_probs,
    temp_bucket,
)
# `_to_internal` is private to runtime.py, and importing it is the lesser evil: the alternative
# is a second copy of laya's question-normalisation rules living in the benchmark, free to drift
# away from the one the runtime actually uses -- at which point this module would be measuring a
# slightly different model than the one that ships.
from ..runtime import _to_internal           # noqa: E402
from ..sequence import build_sequence, render_options   # noqa: E402

DEFAULT_BINS = 15


# --------------------------------------------------------------------------- collection
def collect_logits(agent, dataset: Iterable[Dict[str, Any]],
                   progress_every: int = 0) -> List[Dict[str, Any]]:
    """Run `agent`'s graph over `dataset` and keep the raw per-question logits.

    Raw logits, not the answer dicts, for two reasons. First, `build_answers` rounds its
    published probabilities to 4 decimal places, which for a 77-way choice sends most of the
    tail to exactly 0.0 and makes the NLL that `refit_temps` minimises infinite. Second, a
    temperature refit has to be evaluated at many temperatures, and re-running a 322M-parameter
    encoder once per candidate temperature would make the search cost hours instead of
    milliseconds -- collect once, rescale in numpy.

    The returned rows are one per *question*, not per record.
    """
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(dataset):
        state, questions, gold = rec["state"], rec["questions"], rec["gold"]
        ids = list(questions.keys())

        items, internal = [], {}
        for qid in ids:
            q = _to_internal(questions[qid])
            internal[qid] = q
            seq, markers = build_sequence(agent.tok, state, q, agent.max_len, agent.head_max_len)
            # Mirrors OnnxAgent.system_one's two guards. They matter more here than there: a
            # silently truncated option list would drop the gold label out of the candidate set
            # for some examples and not others, and the resulting accuracy and ECE would be
            # measurements of the truncation, not of the model.
            if len(markers) != len(render_options(q)):
                raise ValueError(
                    "question %r on record %d has more options than head_max_len=%d can hold "
                    "(%d of %d rendered). Measuring a truncated option set would silently drop "
                    "the gold label for some examples." %
                    (qid, i, agent.head_max_len, len(markers), len(render_options(q))))
            if len(markers) < 2:
                raise ValueError("question %r on record %d has %d option(s); the exported graph "
                                 "requires at least 2." % (qid, i, len(markers)))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        batch = collate(items, agent.tok.pad_token_id)
        logits, _act = agent.session.run(batch)

        for r, qid in enumerate(ids):
            k = len(items[r]["markers"])
            qt = QTYPES[internal[qid]["t"]]
            g = int(gold[qid])
            if not 0 <= g < k:
                raise ValueError("record %d question %r: gold index %d is outside the %d "
                                 "options" % (i, qid, g, k))
            rows.append({
                "bucket": temp_bucket(qt, k),
                "qtype": qt,
                "k": k,
                # float64: the fit minimises an NLL over these and float32 accumulation noise is
                # the same order as the int8-vs-fp32 differences being measured.
                "logits": np.asarray(logits[r, :k], dtype=np.float64),
                "gold": g,
            })

        if progress_every and (i + 1) % progress_every == 0:
            print("    ... %d records" % (i + 1), flush=True)

    return rows


# --------------------------------------------------------------------------- metrics
def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def temperature_for(row: Dict[str, Any], temperature: Sequence[float],
                    temperature_by_options: Dict[str, float]) -> float:
    """The temperature `build_answers` would apply to this row, clamp included.

    Resolved exactly as `postprocess.build_answers` resolves it -- bucket override first, the
    per-qtype default second, `clamp_temperature` over whichever won. Anything else here would
    measure a model that does not ship.
    """
    return clamp_temperature(
        temperature_by_options.get(row["bucket"], temperature[row["qtype"]]))


def bucket_metrics(rows: Sequence[Dict[str, Any]], temperature: Sequence[float],
                   temperature_by_options: Dict[str, float],
                   bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Accuracy / mean confidence / ECE / NLL for one bucket's rows, at a given temperature."""
    if not rows:
        return {"n": 0}

    t = temperature_for(rows[0], temperature, temperature_by_options)
    top_conf, pub_conf, correct, nll = [], [], [], []
    for row in rows:
        p = _softmax(row["logits"] / t)
        pred = int(p.argmax())
        correct.append(float(pred == row["gold"]))
        top_conf.append(float(p.max()))
        pub_conf.append(confidence_from_probs(p, row["k"]) if row["qtype"] != QTYPES["noul"]
                        else max(float(p[1]), 1.0 - float(p[1])))
        nll.append(-math.log(max(float(p[row["gold"]]), 1e-12)))

    top_conf = np.asarray(top_conf)
    pub_conf = np.asarray(pub_conf)
    correct = np.asarray(correct)
    return {
        "n": len(rows),
        "k": rows[0]["k"],
        "temperature": round(float(t), 4),
        "accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(top_conf.mean()), 4),
        "ece": round(ece_score(top_conf, correct, bins=bins), 4),
        "mean_published_confidence": round(float(pub_conf.mean()), 4),
        "ece_published_confidence": round(ece_score(pub_conf, correct, bins=bins), 4),
        "nll": round(float(np.mean(nll)), 4),
    }


def metrics_from_rows(rows: Sequence[Dict[str, Any]], temperature: Sequence[float],
                      temperature_by_options: Optional[Dict[str, float]] = None,
                      bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Per-bucket metrics for already-collected rows. Split out from `measure_ece` so the same
    forward passes can be scored at several temperature mappings (before and after a refit)
    without paying for the encoder again."""
    temperature_by_options = temperature_by_options or {}
    out: Dict[str, Any] = {}
    for bucket in sorted({r["bucket"] for r in rows}):
        out[bucket] = bucket_metrics([r for r in rows if r["bucket"] == bucket],
                                     temperature, temperature_by_options, bins=bins)
    return out


def measure_ece(agent, dataset: Iterable[Dict[str, Any]],
                temperature_by_options: Optional[Dict[str, float]] = None,
                bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Measure per-bucket calibration of `agent` over `dataset`.

    Returns `{bucket: {"n", "k", "temperature", "accuracy", "mean_confidence", "ece",
    "mean_published_confidence", "ece_published_confidence", "nll"}}`.

    `temperature_by_options` overrides whatever the agent loaded from its config, which is how
    a candidate refit is scored on held-out data before it is written anywhere.
    """
    rows = collect_logits(agent, dataset)
    return metrics_from_rows(
        rows, agent.temperature,
        agent.temperature_by_options if temperature_by_options is None else temperature_by_options,
        bins=bins)


# --------------------------------------------------------------------------- uncertainty
# An ECE quoted to 4 decimal places on 150 rows across 15 bins invites a comparison the sample
# size does not support. "int8 ECE is within 0.02 of fp32" is a claim about two noisy estimates,
# and in a repository whose rule is that claims are measured or attributed, the noise has to be
# measured too. Two estimators below: a bootstrap interval for any per-bucket statistic, and
# McNemar for the paired accuracy comparison that the ship/don't-ship decision actually rests on.


def correctness(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """0/1 correctness per row. Temperature-free on purpose: a temperature divides every logit
    by the same scalar and cannot reorder them, so accuracy is invariant under it (asserted in
    tests/test_onnx_calibration.py). Taking a temperature here would imply otherwise."""
    return np.asarray([float(int(r["logits"].argmax()) == r["gold"]) for r in rows])


def accuracy_of(rows: Sequence[Dict[str, Any]]) -> float:
    return float(correctness(rows).mean()) if len(rows) else float("nan")


def ece_of(rows: Sequence[Dict[str, Any]], t: float, bins: int = DEFAULT_BINS) -> float:
    """Top-1-probability ECE for `rows` at a fixed temperature `t`."""
    if not len(rows):
        return float("nan")
    conf = np.asarray([float(_softmax(r["logits"] / t).max()) for r in rows])
    return ece_score(conf, correctness(rows), bins=bins)


def bootstrap_ci(rows: Sequence[Dict[str, Any]], statistic, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval for `statistic(rows)`, resampling rows with replacement.

    The bootstrap rather than a closed form because ECE has no usable analytic standard error:
    it is a sum over occupied bins of |mean confidence - mean accuracy| weighted by bin mass,
    and both the bin occupancy and the within-bin means are random. Resampling the rows
    reproduces exactly the sampling variation the reported number is subject to.

    Deterministic from `seed` so a published interval can be reproduced.

    **These intervals are not symmetric about the point estimate, and for ECE they lean high.**
    ECE is a weighted mean of |mean confidence - mean accuracy| over occupied bins: it is a sum
    of absolute values, bounded below by 0 and unbounded above, so resampling can inflate a bin
    gap much further than it can cancel one. In every bucket of this repository's study the
    point estimate sits at or just above the lower bound. Read the upper end as "how bad could
    this be", not as a symmetric error bar, and do not infer a standard error by halving the
    width.
    """
    if not len(rows):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(rows)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = statistic([rows[j] for j in idx])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def _require_aligned(rows_a: Sequence[Dict[str, Any]], rows_b: Sequence[Dict[str, Any]]) -> None:
    """Both lists must be the same questions in the same order, or every paired statistic is
    comparing unrelated answers to each other."""
    if len(rows_a) != len(rows_b):
        raise ValueError("paired statistics need paired rows: got %d and %d"
                         % (len(rows_a), len(rows_b)))
    for ra, rb in zip(rows_a, rows_b):
        if ra["gold"] != rb["gold"] or ra["bucket"] != rb["bucket"]:
            raise ValueError("rows are not aligned: %r/%r vs %r/%r"
                             % (ra["bucket"], ra["gold"], rb["bucket"], rb["gold"]))


def conf_correct(rows: Sequence[Dict[str, Any]], t: float) -> Tuple[np.ndarray, np.ndarray]:
    """`(top-1 confidence, correctness)` per row at temperature `t` -- the two vectors ECE is a
    function of. Computed once so the resampling loops below are array indexing rather than
    thousands of softmaxes."""
    conf = np.asarray([float(_softmax(r["logits"] / t).max()) for r in rows])
    return conf, correctness(rows)


def paired_ece_gap(rows_a: Sequence[Dict[str, Any]], rows_b: Sequence[Dict[str, Any]],
                   t_a: float, t_b: float, n_boot: int = 4000, n_perm: int = 4000,
                   alpha: float = 0.05, seed: int = 0,
                   bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Is the ECE difference between two graphs distinguishable from zero on these rows?

    **Paired, and at full sample size.** Both graphs answered the same questions, so their ECEs
    are computed from the same rows and are strongly positively correlated: a row that lands in
    an over-confident bin does so for both. The variance of the *difference* is therefore
    `Var_a + Var_b - 2 Cov(a, b)`, and an estimator that ignores the covariance term -- for
    instance by drawing two independent bootstrap samples, one per graph -- reports a spread
    close to `sqrt(2)` times the standard error of a single ECE when the true spread is much
    smaller. That overstates the noise, which makes a real difference look like sampling error.
    It is the conservative direction, and it is still wrong: it is also exactly the mistake of
    using a paired test (McNemar) for accuracy and an unpaired null for ECE in the same
    analysis.

    So both procedures below resample *row indices once* and score both graphs on the same
    resampled rows, which respects the pairing the way McNemar's discordant-pair counting does.

    Two complementary answers, because they ask slightly different questions:

    - `gap_ci`: a percentile bootstrap interval for the signed difference `ECE_a - ECE_b`.
      `ci_excludes_zero` is the direct answer to "is this gap real".
    - `null_gap_p95` / `permutation_p`: a permutation null in which each row's two
      `(confidence, correctness)` pairs are exchanged with probability 1/2, which is the
      hypothesis that the two graphs are interchangeable row by row. `null_gap_p95` is the
      smallest |gap| that would be surprising at this `n`, i.e. a "noise floor" that can be
      quoted next to an observed gap.

    The temperature travels with its graph through both procedures: swapping a row means
    swapping the `(conf, correct)` pair that graph's own fitted temperature produced, not
    re-scoring one graph's logits at the other's temperature.
    """
    _require_aligned(rows_a, rows_b)
    if not len(rows_a):
        raise ValueError("cannot compare ECE on zero rows")

    ca, ra_ = conf_correct(rows_a, t_a)
    cb, rb_ = conf_correct(rows_b, t_b)
    ece_a = ece_score(ca, ra_, bins=bins)
    ece_b = ece_score(cb, rb_, bins=bins)
    observed = ece_a - ece_b

    n = len(rows_a)
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)          # one index draw, used for BOTH graphs
        boot[i] = (ece_score(ca[idx], ra_[idx], bins=bins)
                   - ece_score(cb[idx], rb_[idx], bins=bins))
    lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))

    null = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        pa_c, pa_r = np.where(swap, cb, ca), np.where(swap, rb_, ra_)
        pb_c, pb_r = np.where(swap, ca, cb), np.where(swap, ra_, rb_)
        null[i] = abs(ece_score(pa_c, pa_r, bins=bins) - ece_score(pb_c, pb_r, bins=bins))
    # +1 in numerator and denominator: the observed assignment is itself one of the possible
    # permutations, and omitting it can produce p = 0, which no finite permutation test can
    # actually justify.
    perm_p = float((1 + int((null >= abs(observed) - 1e-12).sum())) / (n_perm + 1))

    return {
        "n": n,
        "ece_a": round(float(ece_a), 4),
        "ece_b": round(float(ece_b), 4),
        "gap": round(float(observed), 4),
        "gap_ci": [round(lo, 4), round(hi, 4)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "null_gap_p95": round(float(np.quantile(null, 0.95)), 4),
        "permutation_p": round(perm_p, 4),
    }


def paired_mcnemar(rows_a: Sequence[Dict[str, Any]],
                   rows_b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact two-sided McNemar test on the paired correctness of two graphs over the same rows.

    McNemar rather than a two-sample proportion test because the rows are *paired*: both graphs
    answered the same questions, so the two accuracies are strongly dependent and an unpaired
    test would badly overstate the variance. Only the discordant pairs carry information --
    `b` = a right / b wrong, `c` = a wrong / b right -- and under the null they split 50/50.

    The exact binomial p-value rather than the chi-square approximation: `b + c` here is small
    enough (tens) that the continuity-corrected chi-square is an approximation to something
    `math.comb` can compute outright, and a p-value is exactly the kind of number this task is
    not allowed to approximate for convenience.
    """
    # Pairing is by position; a mismatched gold or bucket means the two lists are not the same
    # questions in the same order, which would make every number below meaningless.
    _require_aligned(rows_a, rows_b)

    ca, cb = correctness(rows_a), correctness(rows_b)
    b = int(((ca == 1) & (cb == 0)).sum())
    c = int(((ca == 0) & (cb == 1)).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2.0 ** n))
    return {"n_pairs": len(rows_a), "b_only_a_right": b, "c_only_b_right": c,
            "discordant": n, "p_value": p,
            "accuracy_a": round(float(ca.mean()), 4), "accuracy_b": round(float(cb.mean()), 4)}


# --------------------------------------------------------------------------- fit/report split
def split_rows(rows: Sequence[Dict[str, Any]], seed: int = 0, min_per_bucket: int = 20
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Split `rows` per bucket into (fit, report, unfittable_buckets).

    A temperature fitted and then evaluated on the same examples is not a calibration result,
    it is a curve fit -- and it reads as a success, which is what makes it dangerous. So the
    split is enforced here rather than left to the caller, per bucket (a global split can leave
    a small bucket entirely on one side), and deterministically from `seed` so a reported number
    can be reproduced.

    A bucket with fewer than `min_per_bucket` rows is returned in the third list and contributes
    nothing to the fit: its rows go to the report half only. Reporting such a bucket as
    *unfitted* is the honest outcome; fitting it on everything it has would produce a
    temperature that has never been tested.
    """
    fit: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    unfittable: List[str] = []
    for bucket in sorted({r["bucket"] for r in rows}):
        group = [r for r in rows if r["bucket"] == bucket]
        if len(group) < min_per_bucket:
            unfittable.append(bucket)
            report.extend(group)
            continue
        order = list(range(len(group)))
        random.Random("%s/%d" % (bucket, seed)).shuffle(order)
        half = len(order) // 2
        fit.extend(group[i] for i in order[:half])
        report.extend(group[i] for i in order[half:])
    return fit, report, unfittable


# --------------------------------------------------------------------------- __main__
def _build_dataset(n_per_dataset: int, seed: int = 13) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch the labelled suites. Lives in `__main__` territory, never in the library functions.

    The five sets are chosen to populate *different* `temp_bucket`s -- a temperature is per
    bucket, so a calibration study that only ever sees 2-way questions has measured one ninth of
    the surface. Loaders mirror `research/scripts/bench_apps.py`'s, but are re-stated here
    rather than imported: `research/` is not importable by this package by repository rule, and
    that harness builds torch-path cases with a different record shape anyway.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    out: Dict[str, List[Dict[str, Any]]] = {}

    # choice:3-5 -- ag_news, 4-way topic.
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    out["ag_news"] = [
        {"state": {"article": r["text"]},
         "questions": {"topic": {"type": "choice",
                                 "instructions": "What is the topic of `article`?",
                                 "criteria": dict(crit)}},
         "gold": {"topic": int(r["label"])}}
        for r in list(d)[:n_per_dataset]]
    # gold is an index into the criteria dict's order, so that order must be ag_news's own
    # label order. Getting this wrong would not raise anywhere -- it would just report a
    # permuted accuracy, which is the kind of quiet wrongness this whole study is about.
    assert keys == ["world", "sports", "business", "sci_tech"]

    # choice:6-10 -- dair-ai/emotion, 6-way.
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out["emotion"] = [
        {"state": {"text": r["text"]},
         "questions": {"emotion": {"type": "choice",
                                   "instructions": "Which emotion is most strongly expressed in `text`?",
                                   "criteria": {n: None for n in names}}},
         "gold": {"emotion": int(r["label"])}}
        for r in list(d)[:n_per_dataset]]

    # choice:11+ -- banking77. The full 77-label option list does not fit in this checkpoint's
    # 256-token option budget, and a truncated option list would silently drop the gold label,
    # so each case is asked against a fixed-size candidate subset that always contains the gold
    # label plus randomly drawn distractors. That is a materially easier task than the 77-way
    # one and its accuracy must not be compared to the published banking77 numbers -- but the
    # bucket it lands in (choice:11+) and the calibration behaviour of a high-cardinality choice
    # are what this study needs, and both are preserved.
    d = load_dataset("mteb/banking77", split="test")
    labels = sorted({x.replace("_", " ") for x in d["label_text"]})
    n_cand = int(os.environ.get("LAYA_BANKING_CANDIDATES", "20"))
    cases = []
    for r in list(d)[:n_per_dataset]:
        goldlab = r["label_text"].replace("_", " ")
        pool = [x for x in labels if x != goldlab]
        chosen = sorted(rng.sample(pool, n_cand - 1) + [goldlab])
        cases.append({"state": {"message": r["text"]},
                      "questions": {"intent": {"type": "choice",
                                               "instructions": "Which banking intent does `message` express?",
                                               "criteria": {c: None for c in chosen}}},
                      "gold": {"intent": chosen.index(goldlab)}})
    out["banking77"] = cases

    # noul:2 -- enron spam.
    d = load_dataset("SetFit/enron_spam", split="test")
    out["enron_spam"] = [
        {"state": {"subject": r.get("subject") or "", "body": (r.get("message") or "")[:3000]},
         "questions": {"is_spam": {"type": "noul",
                                   "instructions": "Is this email unsolicited spam or bulk marketing?"}},
         "gold": {"is_spam": int(r["label"])}}
        for r in list(d)[:n_per_dataset]]

    # noul:2 -- lmsys/toxic-chat jailbreaking, held out of laya's training.
    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    rows = [r for r in d if (r.get("user_input") or "").strip()]
    jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][:n_per_dataset // 2]
    nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][:n_per_dataset - len(jb)]
    mix = jb + nj
    rng.shuffle(mix)
    out["toxic_chat"] = [
        {"state": {"prompt": r["user_input"][:3000]},
         "questions": {"jailbreak": {
             "type": "noul",
             "instructions": "Does `prompt` try to make an AI assistant ignore its rules, "
                             "policies or system instructions?"}},
         "gold": {"jailbreak": int(r["jailbreaking"])}}
        for r in mix]

    return out


def _table(title: str, per_bucket: Dict[str, Any]) -> None:
    print("\n%s" % title)
    print("  %-14s %5s %5s %6s %8s %9s %7s %9s %7s"
          % ("bucket", "n", "k", "temp", "accuracy", "mean_conf", "ECE", "pub_conf", "ECE_pub"))
    for b, m in sorted(per_bucket.items()):
        print("  %-14s %5d %5d %6.3f %8.4f %9.4f %7.4f %9.4f %7.4f"
              % (b, m["n"], m["k"], m["temperature"], m["accuracy"], m["mean_confidence"],
                 m["ece"], m["mean_published_confidence"], m["ece_published_confidence"]))


def main(argv=None) -> int:
    """Measure fp32, measure int8, refit against int8, re-measure -- all on one fixed split.

    Usage: python -m laya_onnx.bench.eval_ece <fp32_dir> <int8_dir> [n_per_dataset] [out.json]
    """
    import argparse

    from laya_onnx import load
    from laya_onnx.export.refit_temps import fit_temperatures

    ap = argparse.ArgumentParser(description="fp32 vs int8 calibration study")
    # Optional, because --rows-in makes the whole study reproducible with no checkpoint at all.
    ap.add_argument("fp32_dir", nargs="?")
    ap.add_argument("int8_dir", nargs="?")
    ap.add_argument("--n", type=int, default=200, help="examples per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    # Collecting 1500 records through two 322M-parameter graphs takes tens of minutes and 1.3 GB
    # of weights on disk. The rows are the only thing any statistic here needs, and they are
    # deterministic, so caching them makes every number below re-derivable from this driver on a
    # machine that has no checkpoint at all -- which is the difference between a published
    # statistic that can be checked and one that has to be taken on trust.
    ap.add_argument("--rows-out", default=None, help="cache the collected rows to this .npz")
    ap.add_argument("--rows-in", default=None,
                    help="load rows from a .npz written by --rows-out instead of running the "
                         "graphs (needs no weights)")
    ap.add_argument("--rows-keys", default="fp32,int8",
                    help="the two array names to read from --rows-in")
    a = ap.parse_args(argv)

    if not a.rows_in and not (a.fp32_dir and a.int8_dir):
        ap.error("give both fp32_dir and int8_dir, or --rows-in to work from cached rows")

    result: Dict[str, Any] = {"n_per_dataset": a.n, "seed": a.seed}

    if a.rows_in:
        ka, kb = a.rows_keys.split(",")
        z = np.load(a.rows_in, allow_pickle=True)
        rows32, rows8 = list(z[ka]), list(z[kb])
        print("loaded %d + %d rows from %s (keys %s, %s)"
              % (len(rows32), len(rows8), a.rows_in, ka, kb))
        # The cache holds rows, not configs. These are the checkpoint defaults, and
        # `laya-multilingual` ships no fitted temperatures at all, so nothing is lost.
        temp32 = temp8 = [1.0, 1.0, 1.0]
        before32: Dict[str, float] = {}
        before8: Dict[str, float] = {}
    else:
        print("building datasets (n=%d per dataset) ..." % a.n, flush=True)
        suites = _build_dataset(a.n)
        dataset = [rec for name in sorted(suites) for rec in suites[name]]
        print("  %d records across %d suites" % (len(dataset), len(suites)))
        result["suites"] = {k: len(v) for k, v in suites.items()}

        print("collecting fp32 logits ...", flush=True)
        ag32 = load(a.fp32_dir)
        rows32 = collect_logits(ag32, dataset, progress_every=100)

        print("collecting int8 logits ...", flush=True)
        ag8 = load(a.int8_dir)
        rows8 = collect_logits(ag8, dataset, progress_every=100)

        temp32, temp8 = ag32.temperature, ag8.temperature
        before32, before8 = ag32.temperature_by_options, ag8.temperature_by_options
        if a.rows_out:
            np.savez_compressed(a.rows_out, fp32=np.array(rows32, dtype=object),
                                int8=np.array(rows8, dtype=object))
            print("cached rows to %s" % a.rows_out)

    fit32, rep32, unfit32 = split_rows(rows32, seed=a.seed)
    fit8, rep8, unfit8 = split_rows(rows8, seed=a.seed)

    # The split is derived from the row ordering, which is identical for both graphs (the same
    # dataset in the same order), so fp32 and int8 are compared on the same held-out examples.
    assert [r["gold"] for r in rep32] == [r["gold"] for r in rep8]

    result["unfitted_buckets"] = sorted(set(unfit32) | set(unfit8))
    result["fp32_before"] = metrics_from_rows(rep32, temp32, before32)
    result["int8_before"] = metrics_from_rows(rep8, temp8, before8)

    # Ruling: the fit is done against the graph that ships. Also fit fp32, purely so the
    # comparison after fitting is like-for-like rather than fp32-unfitted vs int8-fitted.
    result["temps_int8"], result["temps_int8_raw"] = fit_temperatures(fit8)
    result["temps_fp32"], result["temps_fp32_raw"] = fit_temperatures(fit32)
    result["fp32_after"] = metrics_from_rows(rep32, temp32, result["temps_fp32"])
    result["int8_after"] = metrics_from_rows(rep8, temp8, result["temps_int8"])

    _table("fp32, before fitting (held-out half)", result["fp32_before"])
    _table("int8, before fitting (held-out half)", result["int8_before"])
    _table("fp32, after fitting (held-out half)", result["fp32_after"])
    _table("int8, after fitting (held-out half)", result["int8_after"])
    # ---------------------------------------------------------------- uncertainty
    # Emitted by the driver, not by an off-tree script, so every published interval and p-value
    # is re-derivable from `python -m laya_onnx.bench.eval_ece` (with --rows-in, without even a
    # checkpoint). A number in a README whose only provenance is a script nobody has is exactly
    # the kind of claim this repository does not make.
    print("")
    print("=== uncertainty on the held-out half ===")
    buckets = sorted({r["bucket"] for r in rep32})
    result["uncertainty"] = {}
    print("  %-13s %4s | %-26s | %-26s" % ("bucket", "n", "accuracy fp32 [95% CI]",
                                           "accuracy int8 [95% CI]"))
    for b in buckets:
        g32 = [r for r in rep32 if r["bucket"] == b]
        g8 = [r for r in rep8 if r["bucket"] == b]
        a32, a8 = accuracy_of(g32), accuracy_of(g8)
        c32 = bootstrap_ci(g32, accuracy_of, seed=1)
        c8 = bootstrap_ci(g8, accuracy_of, seed=1)
        result["uncertainty"][b] = {
            "n": len(g32),
            "accuracy_fp32": round(a32, 4), "accuracy_fp32_ci": [round(x, 4) for x in c32],
            "accuracy_int8": round(a8, 4), "accuracy_int8_ci": [round(x, 4) for x in c8],
        }
        print("  %-13s %4d | %.4f [%.4f, %.4f]     | %.4f [%.4f, %.4f]"
              % (b, len(g32), a32, c32[0], c32[1], a8, c8[0], c8[1]))

    print("")
    print("  ECE gap, PAIRED (same resampled rows score both graphs; the covariance between")
    print("  two ECEs measured on identical rows is large, and dropping it overstates noise):")
    print("  %-13s %4s %9s %9s %8s %-20s %6s %8s %8s"
          % ("bucket", "n", "ECE_fp32", "ECE_int8", "gap", "gap 95% CI", "sig?", "null_p95",
             "perm_p"))
    for b in buckets:
        g32 = [r for r in rep32 if r["bucket"] == b]
        g8 = [r for r in rep8 if r["bucket"] == b]
        gap = paired_ece_gap(g32, g8, result["temps_fp32"][b], result["temps_int8"][b])
        result["uncertainty"][b]["ece_gap"] = gap
        print("  %-13s %4d %9.4f %9.4f %8.4f [%7.4f, %7.4f] %6s %8.4f %8.4f"
              % (b, gap["n"], gap["ece_a"], gap["ece_b"], gap["gap"],
                 gap["gap_ci"][0], gap["gap_ci"][1],
                 "YES" if gap["ci_excludes_zero"] else "no",
                 gap["null_gap_p95"], gap["permutation_p"]))

    print("")
    print("  paired McNemar on accuracy (b = fp32 right / int8 wrong):")
    print("  %-13s %5s %6s %6s %7s %12s" % ("bucket", "n", "b", "c", "disc", "p"))
    result["mcnemar"] = {}
    for b in buckets + ["ALL"]:
        g32 = rep32 if b == "ALL" else [r for r in rep32 if r["bucket"] == b]
        g8 = rep8 if b == "ALL" else [r for r in rep8 if r["bucket"] == b]
        m = paired_mcnemar(g32, g8)
        result["mcnemar"][b] = m
        print("  %-13s %5d %6d %6d %7d %12.3g"
              % (b, m["n_pairs"], m["b_only_a_right"], m["c_only_b_right"],
                 m["discordant"], m["p_value"]))

    # Rail status, not just "was it clamped". A bucket that fitted at 4.9490 against a 5.0
    # bound was never clamped and would print clean under a clamp-only check, but its optimum
    # sits at the edge of the publishable range and the number should not be read as converged.
    from laya_onnx.export.refit_temps import REACHABLE_BUCKETS, rail_status
    print("")
    print("fitted temperatures (raw -> clamped, with rail status):")
    result["rail"] = {}
    for tag in ("fp32", "int8"):
        raw_map, clamped_map = result["temps_%s_raw" % tag], result["temps_%s" % tag]
        result["rail"][tag] = {b: rail_status(raw_map[b]) for b in raw_map}
        for b in sorted(clamped_map):
            status = result["rail"][tag][b]
            print("  %-4s %-14s %8.4f -> %.4f%s"
                  % (tag, b, raw_map[b], clamped_map[b],
                     "" if status == "ok" else "   " + status.upper()))
    railed = sorted("%s/%s" % (tag, b) for tag in result["rail"]
                    for b, st in result["rail"][tag].items() if st != "ok")
    if railed:
        print("  at or near a clamp bound (NOT converged optima): %s" % ", ".join(railed))
    never = [b for b in REACHABLE_BUCKETS if b not in result["temps_fp32"]]
    if never:
        print("  never measured by this study, so they publish a raw unfitted softmax: %s"
              % ", ".join(never))
    if result["unfitted_buckets"]:
        print("\nunfitted (too few held-out rows): %s" % ", ".join(result["unfitted_buckets"]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
