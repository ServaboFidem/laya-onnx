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

import numpy as np

# laya.common drags in torch. That is allowed here and nowhere else under laya_onnx/ (see this
# package's __init__): bench/ is measurement, not the serving path. Reusing the upstream
# estimator rather than rebinning locally is what makes these ECE numbers comparable to the
# ones already published for the torch path.
from laya.common import ece_score

from ..collate import collate
from ..postprocess import QTYPES, clamp_temperature, confidence_from_probs, temp_bucket
# `_to_internal` is private to runtime.py, and importing it is the lesser evil: the alternative
# is a second copy of laya's question-normalisation rules living in the benchmark, free to drift
# away from the one the runtime actually uses -- at which point this module would be measuring a
# slightly different model than the one that ships.
from ..runtime import _to_internal
from ..sequence import build_sequence, render_options

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
    ap.add_argument("fp32_dir")
    ap.add_argument("int8_dir")
    ap.add_argument("--n", type=int, default=200, help="examples per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    a = ap.parse_args(argv)

    print("building datasets (n=%d per dataset) ..." % a.n, flush=True)
    suites = _build_dataset(a.n)
    dataset = [rec for name in sorted(suites) for rec in suites[name]]
    print("  %d records across %d suites" % (len(dataset), len(suites)))

    result: Dict[str, Any] = {"n_per_dataset": a.n, "seed": a.seed,
                              "suites": {k: len(v) for k, v in suites.items()}}

    print("\ncollecting fp32 logits ...", flush=True)
    ag32 = load(a.fp32_dir)
    rows32 = collect_logits(ag32, dataset, progress_every=100)
    fit32, rep32, unfit32 = split_rows(rows32, seed=a.seed)

    print("collecting int8 logits ...", flush=True)
    ag8 = load(a.int8_dir)
    rows8 = collect_logits(ag8, dataset, progress_every=100)
    fit8, rep8, unfit8 = split_rows(rows8, seed=a.seed)

    # The split is derived from the row ordering, which is identical for both graphs (the same
    # dataset in the same order), so fp32 and int8 are compared on the same held-out examples.
    assert [r["gold"] for r in rep32] == [r["gold"] for r in rep8]

    result["unfitted_buckets"] = sorted(set(unfit32) | set(unfit8))
    result["fp32_before"] = metrics_from_rows(rep32, ag32.temperature, ag32.temperature_by_options)
    result["int8_before"] = metrics_from_rows(rep8, ag8.temperature, ag8.temperature_by_options)

    # Ruling: the fit is done against the graph that ships. Also fit fp32, purely so the
    # comparison after fitting is like-for-like rather than fp32-unfitted vs int8-fitted.
    result["temps_int8"], result["temps_int8_raw"] = fit_temperatures(fit8)
    result["temps_fp32"], result["temps_fp32_raw"] = fit_temperatures(fit32)
    result["fp32_after"] = metrics_from_rows(rep32, ag32.temperature, result["temps_fp32"])
    result["int8_after"] = metrics_from_rows(rep8, ag8.temperature, result["temps_int8"])

    _table("fp32, before fitting (held-out half)", result["fp32_before"])
    _table("int8, before fitting (held-out half)", result["int8_before"])
    _table("fp32, after fitting (held-out half)", result["fp32_after"])
    _table("int8, after fitting (held-out half)", result["int8_after"])
    print("\nfitted temperatures (raw -> clamped):")
    for b in sorted(result["temps_int8"]):
        print("  int8 %-14s %8.4f -> %.4f%s"
              % (b, result["temps_int8_raw"][b], result["temps_int8"][b],
                 "   CLAMPED" if abs(result["temps_int8_raw"][b] - result["temps_int8"][b]) > 1e-9
                 else ""))
    for b in sorted(result["temps_fp32"]):
        print("  fp32 %-14s %8.4f -> %.4f%s"
              % (b, result["temps_fp32_raw"][b], result["temps_fp32"][b],
                 "   CLAMPED" if abs(result["temps_fp32_raw"][b] - result["temps_fp32"][b]) > 1e-9
                 else ""))
    if result["unfitted_buckets"]:
        print("\nunfitted (too few held-out rows): %s" % ", ".join(result["unfitted_buckets"]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
