"""Wall-clock latency of `OnnxAgent.system_one`, as a function of questions per call.

Why this script exists, and why it is shaped the way it is:

1. **The first call is not a measurement.** `OnnxSession.__init__` builds the session with
   `ORT_ENABLE_ALL`, but onnxruntime defers most of that work -- operator fusion, memory-arena
   sizing, thread-pool spin-up -- until the first `run()` with a given input *shape*. Because
   all three axes of this graph are dynamic (batch, sequence, markers), a call with 5 questions
   pays that cost again after a call with 1 did. So `measure()` takes a `warmup` count and
   discards those runs, and the default of 5 is not arbitrary: on the machine in the README's
   table, run 1 at 1 question cost roughly 3x run 6, and runs 2-5 were already flat.

2. **p50 and p95, not mean.** A mean over a CPU inference loop is dominated by whatever else the
   host was doing; the tail is the number a caller actually has to size a timeout against. 50
   runs is the floor for a p95 to mean anything at all (it is the 3rd-slowest sample), and is
   what the default reports -- treat a p95 from 50 runs as indicative, not tight.

3. **Questions per call, not batch size.** laya's whole premise is that N typed questions about
   one state cost one forward pass, so the interesting curve is latency vs. N. `collate()` pads
   every question in a call to the longest sequence in it, and each question carries its own
   copy of the state, so N questions is genuinely an N-row batch -- the curve is expected to
   rise, and the point of measuring is to find out how steeply.

4. **Torch-free by construction.** The questions below are written out here rather than imported
   from `laya.presets`, even though that module is itself dependency-free, because
   `import laya.presets` executes `laya/__init__.py`, which imports torch. A latency number for
   the ONNX path measured in a process that has torch resident is not the number a deployment
   gets. This module imports `laya_onnx` and the stdlib, nothing else.

`bench/`'s no-fetch rule (see `laya_onnx/bench/__init__.py`) holds here too: the state and the
questions are literals in this file, so the script runs offline against a local export.

Run it:

    python -m laya_onnx.bench.bench_latency ~/laya_onnx_models/multilingual
    python -m laya_onnx.bench.bench_latency <dir> --runs 100 --threads 8 --counts 1,5,10,50
"""
import argparse
import os
import time
from typing import Any, Dict, List, Sequence

# transformers is not imported anywhere in this path, but a caller may have set neither var and
# some other library in the process may probe them; keep the repo-wide guards on every entry
# point regardless (see AGENTS.md).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# A realistic support-ticket state: long enough to exercise the sequence builder, short enough
# that every question in the set fits the 1024-token budget without truncation, so the numbers
# measure inference and not the truncation path.
STATE = {
    "subject": "Charged twice for the Pro plan this month",
    "from": "dana.whitfield@northline.example",
    "body": (
        "Hi -- I upgraded to Pro on the 3rd and the card was charged 49.00 that day, which is "
        "correct. On the 5th the same card was charged 49.00 again, same descriptor, same "
        "amount. I checked my invoices page and I only see one invoice. This is the second "
        "billing problem in three months and I am running payroll off this account, so I need "
        "the duplicate reversed before Friday or I will have to move to a different provider. "
        "Can someone look at this today? Ticket numbers from the last time were 41822 and 41907."
    ),
    "plan": "pro",
    "account_age_days": 412,
}

# Six distinct questions across all three qtypes and several option counts. `questions(n)` cycles
# these and suffixes the id, so a 50-question call is 50 *distinct* entries in the dict (laya
# keys answers by question id) that still spread across the qtypes rather than being one question
# repeated -- a 50-way call of nothing but `noul` would flatter the numbers, since noul builds the
# shortest option head of the three.
_TEMPLATES: List[Dict[str, Any]] = [
    {
        "type": "choice",
        "instructions": "What does the customer want in `body`?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "billing_question": "a question about an invoice, plan or payment method",
            "information": "general information, pricing or how-to",
            "cancellation": "wants to cancel or downgrade",
            "other": "none of the other options fits",
        },
    },
    {"type": "noul", "instructions": "Does `body` communicate time pressure or a deadline?"},
    {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `body`?",
        "criteria": [
            "calm and neutral",
            "concerned but civil",
            "clearly annoyed",
            "very angry or using strong language",
        ],
    },
    {"type": "noul", "instructions": "Does the customer ask for money back?"},
    {
        "type": "choice",
        "instructions": "Which team should handle the email in `body`?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, demos, new purchases",
            "other": "none of the above",
        },
    },
    {"type": "noul", "instructions": "Does `body` suggest the customer may leave for a competitor?"},
]


def questions(n: int) -> Dict[str, Dict[str, Any]]:
    """`n` distinct question definitions, cycling the templates above in a fixed order."""
    if n < 1:
        raise ValueError("n must be >= 1, got %d" % n)
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(n):
        out["q%02d" % i] = dict(_TEMPLATES[i % len(_TEMPLATES)])
    return out


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on an already-collected sample.

    Nearest-rank rather than interpolated on purpose: every value reported is then a latency that
    was actually observed, which is what you want from a tail statistic. `p` is a fraction, so
    p95 is `percentile(xs, 0.95)`.
    """
    if not samples:
        raise ValueError("no samples")
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), int(-(-p * len(ordered) // 1))))  # ceil without math.ceil
    return ordered[rank - 1]


def measure(agent, state, qs: Dict[str, Dict[str, Any]], runs: int = 50, warmup: int = 5) -> Dict[str, float]:
    """Time `agent.system_one(state, qs)` `runs` times after `warmup` discarded runs.

    `perf_counter` rather than `process_time`: onnxruntime does the work on its own thread pool,
    so process CPU time would count every worker thread and wall clock is what a caller waits.
    """
    for _ in range(warmup):
        agent.system_one(state, qs)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        agent.system_one(state, qs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n_questions": len(qs),
        "runs": runs,
        "warmup": warmup,
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", help="a directory produced by laya_onnx.export.export_fp32")
    ap.add_argument("--runs", type=int, default=50, help="timed runs per question count (default 50)")
    ap.add_argument("--warmup", type=int, default=5, help="discarded runs per question count (default 5)")
    ap.add_argument("--threads", type=int, default=None,
                    help="onnxruntime intra_op_num_threads; default leaves ORT's own default, "
                         "which is every physical core and is usually wrong on a shared host")
    ap.add_argument("--counts", default="1,5,10,50", help="comma-separated questions-per-call (default 1,5,10,50)")
    args = ap.parse_args(argv)

    from laya_onnx import load  # imported here so --help works without onnxruntime installed

    agent = load(os.path.expanduser(args.model_dir), threads=args.threads)
    counts = [int(c) for c in args.counts.split(",") if c.strip()]

    print("model_dir : %s" % os.path.expanduser(args.model_dir))
    print("threads   : %s" % ("ORT default" if args.threads is None else args.threads))
    print("budget    : max_len=%d head_max_len=%d" % (agent.max_len, agent.head_max_len))
    print()
    print("%10s %10s %10s %10s %10s" % ("questions", "p50 ms", "p95 ms", "min ms", "max ms"))
    rows = []
    for n in counts:
        r = measure(agent, STATE, questions(n), runs=args.runs, warmup=args.warmup)
        rows.append(r)
        print("%10d %10.1f %10.1f %10.1f %10.1f"
              % (r["n_questions"], r["p50_ms"], r["p95_ms"], r["min_ms"], r["max_ms"]))

    # Truncation is reported once rather than per row: the state is fixed, so if it fits at 1
    # question it fits at 50, and a reader should be able to see that these numbers are not a
    # measurement of the truncation path.
    rep = agent.system_one(STATE, questions(max(counts)))["usage"]["truncated"]
    dropped = {q: r for q, r in rep.items() if r.get("truncated")}
    print("\ntruncated questions: %d of %d" % (len(dropped), len(rep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
