"""End-to-end parity between the torch path and the ONNX path. Real weights, real forwards.

Run:  python tests/test_onnx_local_e2e.py [torch_model_dir] [onnx_model_dir]

Defaults to ~/laya_models/laya-multilingual and ~/laya_onnx_models/multilingual. NOT in CI --
it needs a 644 MB torch checkpoint and a 1.3 GB ONNX export on local disk, which is exactly
why tests/test_local_e2e.py is excluded too. Everything CI runs is fixture-backed and offline.

Produce the ONNX side with:

    python -c "from huggingface_hub import snapshot_download; \\
               print(snapshot_download('convaiinnovations/laya', allow_patterns=['multilingual/*']))"
    python -m laya_onnx.export.export_fp32 \\
        --model-dir <snapshot>/multilingual \\
        --out ~/laya_onnx_models/multilingual

That writes FOUR things, not three. Alongside `rl_agent_config.json` and `tokenizer/` there
is `model.onnx` (~2.8 MB, the graph) and `model.onnx.data` (~1.29 GB, every weight). torch
splits them because an fp32 mmBERT export is far past protobuf 2 GB message ceiling. They
are one artifact in two files: copy `model.onnx` alone and you have deployed a graph with no
weights in it. The exporter names the `.data` file on stdout and refuses to finish if it is
missing, so trust that summary over this docstring.

Opset 18 is the default, and 17 is not an option here. torch.export captures at 18 and asks a converter to walk the graph back
down; on mmBERT that converter relabels the header to 17 and leaves a `Split(num_outputs=2)`
node behind, producing a file that declares 17, does not validate against 17, and is rejected
by onnxruntime at session creation with INVALID_GRAPH. `_verify_opset` now catches that at
export time -- see laya_onnx/export/export_fp32.py.

WHAT THIS FILE CHECKS, AND WHY IN THIS ORDER
--------------------------------------------
1. The three dynamic axes, on the real encoder. `dynamic_axes=` with `dynamo=True` is a
   combination torch itself warns about, and a graph that froze its marker axis passes any
   single-question test and fails on the second real question. So this runs the real session
   at shapes the trace never saw -- including combinations that hold two axes at their traced
   values and move only the third, which is the only way to show the axes are independent
   rather than merely all-different-at-once. Nothing downstream is trustworthy until this
   passes, so it runs first.
2. Graph numerics: the torch `DecisionModel` and the ONNX session, fed the *same* collated
   batch, must agree on raw logits. This is the narrow question "did the export change the
   arithmetic", isolated from tokenization and postprocessing.
3. Pipeline parity: `laya.load(...).predict` vs `laya_onnx.load(...).predict` on the same
   states and questions. This is the wide question, and it is the one that catches a
   divergence in sequence construction or special-token resolution rather than in the graph.
   The answer dicts must be *structurally* identical (same question ids, same types, same
   keys, same chosen labels) and numerically within 2e-3.
4. The budget claim. This port runs on mmBERT rather than the English checkpoint for exactly
   one reason: 1024/256 leaves ~768 tokens for the state where 512/192 leaves ~317. That is
   an assertion about real inputs, so it is measured on a real one -- a token-dense code state
   -- rather than assumed.

A NOTE ON LAYA_DEVICE
---------------------
Honored, and defaulted to "cpu" deliberately. laya/agent.py:221-226 picks the compute dtype
from the device: fp32 on cpu/mps, but the checkpoint's `amp_dtype` (bf16 here) on cuda. The
ONNX artifact is fp32. Comparing an fp32 graph against a bf16 torch run measures the dtype
gap, not the export, and would blow past every threshold below for reasons that have nothing
to do with correctness. Set LAYA_DEVICE=cuda if you want to see that number, but read a
failure as "different dtypes disagree", which is not news.
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# transformers probes for TensorFlow at import and its abseil runtime can deadlock model
# construction. Every entry point in this repo sets these before the first transformers import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import laya  # noqa: E402
import laya_onnx  # noqa: E402
from laya_onnx.collate import collate  # noqa: E402
from laya_onnx.postprocess import QTYPES  # noqa: E402
from laya_onnx.runtime import _to_internal  # noqa: E402
from laya_onnx.sequence import build_sequence  # noqa: E402
from laya_onnx.truncation import truncation_report  # noqa: E402

DEVICE = os.environ.get("LAYA_DEVICE", "cpu")
ONNX_DIR = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/laya_onnx_models/multilingual")

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, ("  " + detail) if detail else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def head(t):
    print("\n" + "=" * 78 + "\n  " + t + "\n" + "=" * 78, flush=True)


def resolve_torch_dir():
    """Locate the torch-side multilingual checkpoint.

    argv[1] wins. Otherwise ~/laya_models/laya-multilingual, matching the layout
    tests/test_local_e2e.py already documents. Failing that, fall back to the Hugging Face
    cache, because the natural way to get these weights is `snapshot_download`, which puts
    them there and nowhere else -- and requiring a manual copy into ~/laya_models just to run
    this file would be friction with no payoff. `local_files_only=True` keeps that fallback
    offline: it resolves an already-downloaded snapshot or raises, and never reaches for the
    network behind the runner's back.
    """
    if len(sys.argv) > 1:
        return os.path.expanduser(sys.argv[1])
    home = os.path.expanduser("~/laya_models/laya-multilingual")
    if os.path.isdir(home):
        return home
    from huggingface_hub import snapshot_download
    snap = snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"],
                             local_files_only=True)
    return os.path.join(snap, "multilingual")


TORCH_DIR = resolve_torch_dir()
print("torch checkpoint : %s" % TORCH_DIR)
print("onnx  checkpoint : %s" % ONNX_DIR)
print("device           : %s" % DEVICE)

torch_agent = laya.load(TORCH_DIR, device=DEVICE)
onnx_agent = laya_onnx.load(ONNX_DIR)
print("torch compute dtype: %s" % torch_agent.dtype)

# A token-dense code state: a medium function repeated enough to look like a real file rather
# than a snippet. This is the input the whole checkpoint choice was made for, so it is a
# first-class parity case and not only a truncation probe.
CODE = """def reconcile(ledger, payments):
    seen = {}
    for p in payments:
        if p.invoice_id in seen:
            raise DuplicatePayment(p.invoice_id)
        seen[p.invoice_id] = p
    for entry in ledger:
        p = seen.get(entry.invoice_id)
        if p is None:
            entry.status = "unpaid"
        elif p.amount != entry.amount:
            entry.status = "mismatch"
        else:
            entry.status = "settled"
    return ledger
""" * 3

STATES = [
    {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
     "body": "We were billed twice for March. Refund the duplicate or we cancel."},
    {"code": CODE},
    "Plain prose with no structure at all.",
]
CODE_STATE = STATES[1]
SUITES = {"triage": laya.triage_questions(), "guard": laya.guard_questions(),
          "moderation": laya.moderation_questions(), "router": laya.router_questions()}

# The rounding the answer dicts carry (laya_onnx/postprocess.py rounds to 4dp, as the torch
# path does) means an equality on those values is a weak instrument: two runs differing by
# 1e-5 round identically. So the answer dicts are checked for structural identity and 2e-3
# agreement -- the user-visible contract -- while the *numeric* question is asked separately
# in section 2, against unrounded logits, where a real divergence has nowhere to hide.
PROB_TOL = 2e-3
# The spec's fp32 parity bar, not a bar chosen to fit the measurement. Measured here is
# 3.7e-05, so there is ~27x headroom; the point of pinning it at the spec value rather than
# somewhere comfortably above it is that a future export which degrades to, say, 9e-4 must
# fail this suite instead of passing it while violating the spec.
LOGIT_TOL = 1e-4


def collate_for(state, questions):
    """Build exactly the batch OnnxAgent.system_one would build for these inputs."""
    items = []
    for qdef in questions.values():
        q = _to_internal(qdef)
        seq, markers = build_sequence(onnx_agent.tok, state, q,
                                      onnx_agent.max_len, onnx_agent.head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate(items, onnx_agent.tok.pad_token_id)


# ------------------------------------------------- 1. dynamic axes, on the real encoder
head("1. Dynamic axes on the real export (shapes the trace never saw)")
# The export sample was batch=2, seq=32, markers=3 (laya_onnx/export/export_fp32.py). Each row
# below differs from it, and the last three move exactly one axis while pinning the other two
# at their traced values -- that is what distinguishes "all three axes are symbolic and
# independent" from "the graph happens to tolerate this particular combination".
TRACED = (2, 32, 3)
SHAPES = [
    (1, 7, 2),      # everything smaller; markers at the topk(2) boundary
    (1, 1024, 11),  # the checkpoint's full sequence budget, many options
    (3, 64, 4),
    (7, 5, 31),     # more markers than sequence positions
    (5, 32, 3),     # batch only
    (2, 33, 3),     # sequence only
    (2, 32, 9),     # markers only  <- the axis most likely to have been frozen
]
rng = np.random.default_rng(0)
for b, sq, mk in SHAPES:
    which = [n for n, v, tv in zip(("batch", "seq", "markers"), (b, sq, mk), TRACED) if v != tv]
    batch = {
        "input_ids": rng.integers(1, 1000, (b, sq)).astype(np.int64),
        "attention_mask": np.ones((b, sq), dtype=np.int64),
        "marker_pos": rng.integers(0, sq, (b, mk)).astype(np.int64),
        "marker_mask": np.ones((b, mk), dtype=bool),
        "qtype": np.zeros((b,), dtype=np.int64),
    }
    logits, act = onnx_agent.session.run(batch)
    ok("axes/b%d.s%d.m%d" % (b, sq, mk),
       logits.shape == (b, mk) and act.shape[0] == b,
       "varies %s -> logits%s act%s" % ("+".join(which) or "nothing", logits.shape, act.shape))

# ------------------------------------------------- 2. graph numerics on identical inputs
head("2. Graph numerics: torch DecisionModel vs ONNX session, same collated batch")
worst_logit = worst_act = 0.0
for sname, questions in SUITES.items():
    for i, state in enumerate(STATES):
        batch = collate_for(state, questions)
        onnx_logits, onnx_act = onnx_agent.session.run(batch)
        with torch.no_grad():
            args = [torch.as_tensor(batch[k]).to(torch_agent.device)
                    for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            t_logits, t_act = torch_agent.model(*args)
        t_logits = t_logits.float().cpu().numpy()
        t_act = t_act.float().cpu().numpy()
        dl = float(np.abs(t_logits - onnx_logits).max())
        worst_logit = max(worst_logit, dl)
        ok("numerics/%s/s%d/logits" % (sname, i), dl < LOGIT_TOL,
           "shape=%s max|d|=%.3e" % (tuple(batch["input_ids"].shape), dl))
        # act_logits are compared after the softmax that actually consumes them. The raw act
        # head carries a much wider dynamic range than the option logits, so an absolute delta
        # on it is not comparable to `dl` and says little; the probability is what reaches the
        # caller as `action.act_probability`, so that is what gets a tolerance.
        ex = np.exp(t_act - t_act.max(-1, keepdims=True))
        eo = np.exp(onnx_act - onnx_act.max(-1, keepdims=True))
        da = float(np.abs(ex / ex.sum(-1, keepdims=True) - eo / eo.sum(-1, keepdims=True)).max())
        worst_act = max(worst_act, da)
        ok("numerics/%s/s%d/act-probs" % (sname, i), da < PROB_TOL, "max|d|=%.3e" % da)

worst_prob = 0.0


def compare_answers(prefix, t, o):
    """Assert two answer dicts are the same answer, structurally and numerically."""
    global worst_prob
    ok("%s/same-question-ids" % prefix, set(t) == set(o),
       "torch-only=%s onnx-only=%s" % (sorted(set(t) - set(o)), sorted(set(o) - set(t))))
    for qid in sorted(set(t) & set(o)):
        a, b = t[qid], o[qid]
        ok("%s/%s/type" % (prefix, qid), a["type"] == b["type"])
        # Structural identity, not just numeric agreement: a missing "legend" or a stray
        # extra key is a contract break even when every number matches.
        ok("%s/%s/keys" % (prefix, qid), set(a) == set(b),
           "torch-only=%s onnx-only=%s" % (sorted(set(a) - set(b)), sorted(set(b) - set(a))))
        if a["type"] == "choice":
            # A changed argmax on fp32 would mean the export is wrong, not imprecise, so this
            # is an equality and not a tolerance.
            ok("%s/%s/choice" % (prefix, qid), a["choice"] == b["choice"],
               "torch=%r onnx=%r" % (a["choice"], b["choice"]))
            d = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
        elif a["type"] == "score":
            ok("%s/%s/legend" % (prefix, qid), a.get("legend") == b.get("legend"))
            d = abs(a["score"] - b["score"])
        else:
            d = abs(a["noul"] - b["noul"])
        worst_prob = max(worst_prob, d)
        ok("%s/%s/value<%g" % (prefix, qid, PROB_TOL), d < PROB_TOL, "|d|=%.3g" % d)
        ok("%s/%s/confidence<%g" % (prefix, qid, PROB_TOL),
           abs(a["confidence"] - b["confidence"]) < PROB_TOL,
           "|d|=%.3g" % abs(a["confidence"] - b["confidence"]))


# ------------------------------------------------- 3. full-pipeline parity
head("3. Pipeline parity: laya.predict vs laya_onnx.predict")
for sname, questions in SUITES.items():
    for i, state in enumerate(STATES):
        compare_answers("%s/s%d" % (sname, i),
                        torch_agent.predict(state, questions)["answers"],
                        onnx_agent.predict(state, questions)["answers"])

# ------------------------------------------------- 4. non-Latin script
head("4. mmBERT reads non-Latin scripts")
# The whole reason the router exists is that the English checkpoint collapses off English
# while staying confident. This port is the multilingual one, so Devanagari must produce a
# full answer set rather than an exception -- and the ONNX path must agree with torch on it
# just as closely as it does on English. Given the Gemma tokenizer, this is also the case
# where a special-token or byte-level encoding divergence would show up first, which is why
# it gets the same full comparison rather than a smoke test.
DEV = {"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया।"}
dev_o = onnx_agent.predict(DEV, SUITES["triage"])["answers"]
ok("devanagari/answers-every-question", set(dev_o) == set(SUITES["triage"]),
   "got %s" % sorted(dev_o))
compare_answers("devanagari", torch_agent.predict(DEV, SUITES["triage"])["answers"], dev_o)

# ------------------------------------------------- 5. the budget claim, measured
head("5. Token budget: does a real code state fit in 1024/256?")
usage = onnx_agent.predict(CODE_STATE, SUITES["triage"])["usage"]
report = usage["truncated"]
for qid, r in report.items():
    print("   %-18s state_tokens=%-5d used=%-5d dropped=%-5d truncated=%s"
          % (qid, r["state_tokens"], r["state_tokens_used"], r["dropped"], r["truncated"]))
ok("budget/code-state-fits-at-1024",
   not any(r["truncated"] for r in report.values()),
   "dropped: %s" % {k: v["dropped"] for k, v in report.items() if v["truncated"]})

# The same state against the English checkpoint's budget. This is the trade the port was made
# for, stated as a measurement instead of a claim: if this state fits in 512/192 too, then
# 1024/256 bought nothing on this input and the checkpoint choice needs re-arguing.
english_q = _to_internal(SUITES["triage"]["intent"])
seq512, _ = build_sequence(onnx_agent.tok, CODE_STATE, english_q, 512, 192)
r512 = truncation_report(onnx_agent.tok, CODE_STATE, seq512, 512, english_q, 192)
state_tokens = next(iter(report.values()))["state_tokens"]
print("   at 512/192 (the English checkpoint's budget): used=%d dropped=%d truncated=%s"
      % (r512["state_tokens_used"], r512["dropped"], r512["truncated"]))
ok("budget/same-state-would-truncate-at-512", r512["truncated"],
   "state is %d tokens; 1024/256 kept all of it, 512/192 kept %d"
   % (state_tokens, r512["state_tokens_used"]))

# ------------------------------------------------- summary
head("summary")
print("max|delta| raw logits (torch vs onnx) : %.3e   (tol %g)" % (worst_logit, LOGIT_TOL))
print("max|delta| act probabilities          : %.3e   (tol %g)" % (worst_act, PROB_TOL))
print("max|delta| reported answer values     : %.3e   (tol %g)" % (worst_prob, PROB_TOL))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx end-to-end parity tests passed")
sys.exit(1 if FAIL else 0)
