"""The OpenVINO backend: same graph, same answers as onnxruntime, and the properties it pins.

Offline and fixture-backed, like tests/test_onnx_runtime.py: a tiny from-config BERT is wrapped
in the real DecisionModel and exported with the real export_fp32, so both runtimes load the same
kind of graph the shipped export is, just small. The real-weights comparison is
tests/test_onnx_local_e2e.py with LAYA_ONNX_BACKEND=openvino, which is not in CI.

What each block below pins, and why it would matter if it broke:

1. Numerics at shapes the trace never saw, moving batch, sequence and marker axes one at a time.
   OpenVINO re-derives shapes from the ONNX graph itself; a dimension it resolved as static would
   pass a single-shape check and fail on the second real call.
2. End-to-end answers through OnnxAgent with either backend -- the claim callers rely on.
3. fp32 is pinned, read back from the compiled model. On hardware with AMX/AVX512_BF16 the CPU
   plugin would otherwise pick bf16 on its own and quietly serve a different-precision model.
4. `threads` reaches INFERENCE_NUM_THREADS.
5. The missing-sidecar error is the shared, named one -- not OpenVINO's own read failure.
6. An unknown backend is a ValueError before any file is touched.
7. Concurrent calls from several threads on one agent return what serial calls return. This is
   the reason the session keeps one InferRequest per thread instead of calling the compiled model.
8. A process that chose OpenVINO never imports onnxruntime (nor torch), checked in a subprocess
   with an import blocker so a guarded `try: import onnxruntime` cannot hide.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                            # noqa: E402
import onnx                                                   # noqa: E402
import openvino as ov                                         # noqa: E402
from tokenizers import Tokenizer                              # noqa: E402
from tokenizers.models import WordLevel                       # noqa: E402
from tokenizers.pre_tokenizers import Whitespace               # noqa: E402
from transformers import AutoConfig, AutoModel                # noqa: E402

from laya.common import DecisionModel                          # noqa: E402
from laya_onnx.export.export_fp32 import export_fp32           # noqa: E402
from laya_onnx.runtime import BACKENDS, OnnxAgent, load        # noqa: E402
from laya_onnx.session import OnnxSession                      # noqa: E402
from laya_onnx.session_openvino import OpenVinoSession         # noqa: E402

PASS, FAIL = [], []

# fp32 on both sides; the two runtimes differ only in kernel order and fusion. The real-weights
# comparison measured 3.2e-05 at most, so 1e-4 is a bar with room, not a loose one.
TOL = 1e-4
VOCAB = 64


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def make_tokenizer(tokdir):
    os.makedirs(tokdir)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i, w in enumerate(["billing", "refund", "charge", "state", "true", "false", "good", "bad", "ok"]):
        vocab[w] = 5 + i
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(os.path.join(tokdir, "tokenizer.json"))
    with open(os.path.join(tokdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]",
                   "sep_token": "[SEP]", "mask_token": "[MASK]"}, f)


def make_checkpoint(root):
    make_tokenizer(os.path.join(root, "tokenizer"))
    with open(os.path.join(root, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump({"max_len": 96, "head_max_len": 48}, f)
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=64, vocab_size=VOCAB, max_position_embeddings=128)
    model = DecisionModel(AutoModel.from_config(cfg, attn_implementation="sdpa"), head_layers=1, n_act=2)
    model.eval()
    export_fp32(model, os.path.join(root, "model.onnx"))
    return root


def batch(n, seq, markers, seed):
    """A collate()-shaped batch: int64 everywhere but the bool marker_mask, padding on some rows,
    and per-row marker counts between 2 and `markers` so the masked -1e4 slots are exercised."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, VOCAB, size=(n, seq), dtype=np.int64)
    att = np.ones((n, seq), dtype=np.int64)
    mpos = np.zeros((n, markers), dtype=np.int64)
    mmask = np.zeros((n, markers), dtype=bool)
    for i in range(n):
        length = seq if i == 0 else int(rng.integers(max(markers + 2, seq // 2), seq + 1))
        att[i, length:] = 0
        ids[i, length:] = 0
        k = markers if i == 0 else int(rng.integers(2, markers + 1))
        mpos[i, :k] = np.sort(rng.choice(np.arange(1, length), size=k, replace=False))
        mmask[i, :k] = True
    qtype = rng.integers(0, 3, size=(n,), dtype=np.int64)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qtype}


QUESTIONS = {
    "c": {"type": "choice", "instructions": "billing state",
          "criteria": {"good": "good state", "bad": "bad state", "ok": "ok state"}},
    "s": {"type": "score", "instructions": "charge state", "criteria": ["bad", "ok", "good", "good state"]},
    "n": {"type": "noul", "instructions": "refund state"},
}


def answers_close(a, b):
    """Same question ids, same labels, and every published number within 2e-4 -- they are rounded
    to 4 decimals, so identical logits can still differ by one unit in the last place."""
    if a.keys() != b.keys():
        return False, "ids %s vs %s" % (sorted(a), sorted(b))
    for qid in a:
        x, y = a[qid], b[qid]
        if x["type"] != y["type"] or x.get("choice") != y.get("choice"):
            return False, "%s: %r vs %r" % (qid, x, y)
        nums_x = [x.get("score", 0), x.get("noul", 0), x["confidence"]] + list(x.get("probabilities", {}).values())
        nums_y = [y.get("score", 0), y.get("noul", 0), y["confidence"]] + list(y.get("probabilities", {}).values())
        if max(abs(p - q) for p, q in zip(nums_x, nums_y)) > 2e-4:
            return False, "%s: %r vs %r" % (qid, x, y)
    return True, ""


tmp = tempfile.mkdtemp()
try:
    ck = make_checkpoint(tmp)
    path = os.path.join(ck, "model.onnx")
    ort_s, ov_s = OnnxSession(path), OpenVinoSession(path)

    # --- 1. numerics, each axis moved on its own, then all at once ----------------------------
    shapes = [(2, 32, 3), (5, 32, 3), (2, 71, 3), (2, 32, 6), (1, 12, 2), (7, 90, 5)]
    for i, (n, seq, k) in enumerate(shapes):
        b = batch(n, seq, k, seed=i)
        (lo, ao), (lv, av) = ort_s.run(b), ov_s.run(b)
        dl = float(np.abs(lo - lv)[b["marker_mask"]].max())
        da = float(np.abs(ao - av).max())
        tag = "numerics/b%d-s%d-k%d" % (n, seq, k)
        check(tag + "/shape", lo.shape == lv.shape and ao.shape == av.shape, "%s vs %s" % (lo.shape, lv.shape))
        check(tag + "/logits", dl <= TOL, "max|dlogit|=%.3e" % dl)
        check(tag + "/act", da <= TOL, "max|dact|=%.3e" % da)
        check(tag + "/float32", lv.dtype == np.float32 and av.dtype == np.float32)

    # outputs are copies, not views into the request's buffers
    b1, b2 = batch(2, 32, 3, seed=100), batch(2, 32, 3, seed=101)
    first, _ = ov_s.run(b1)
    snapshot = first.copy()
    ov_s.run(b2)
    check("numerics/outputs-are-not-overwritten-by-next-call", np.array_equal(first, snapshot))

    # --- 2. end to end through OnnxAgent ------------------------------------------------------
    a_ort, a_ov = OnnxAgent(ck), OnnxAgent(ck, backend="openvino")
    check("agent/default-backend-is-onnxruntime", a_ort.backend == "onnxruntime", a_ort.backend)
    check("agent/backend-recorded", a_ov.backend == "openvino", a_ov.backend)
    check("agent/openvino-session-class", type(a_ov.session).__name__ == "OpenVinoSession")
    check("agent/load-passes-backend", load(ck, backend="openvino").backend == "openvino")
    for state in ["billing refund charge state", "good " * 60 + "bad ok"]:
        r_ort, r_ov = a_ort.predict(state, QUESTIONS), a_ov.predict(state, QUESTIONS)
        ok, detail = answers_close(r_ort["answers"], r_ov["answers"])
        check("agent/answers-match/%d-chars" % len(state), ok, detail)
        check("agent/usage-match/%d-chars" % len(state), r_ort["usage"] == r_ov["usage"])
        check("agent/model-name/%d-chars" % len(state), r_ov["model"] == "laya-onnx")

    # --- 3 and 4. pinned precision, honoured threads ------------------------------------------
    check("props/precision-is-f32", ov_s._compiled.get_property("INFERENCE_PRECISION_HINT") == ov.Type.f32)
    check("props/latency-hint", str(ov_s._compiled.get_property("PERFORMANCE_HINT")).upper().endswith("LATENCY"),
          str(ov_s._compiled.get_property("PERFORMANCE_HINT")))
    check("props/threads-honoured", OpenVinoSession(path, threads=2).threads == 2)
    check("props/agent-threads-reach-session", OnnxAgent(ck, threads=3, backend="openvino").session.threads == 3)

    # --- 5. the missing-sidecar error is the shared one ---------------------------------------
    ext_dir = os.path.join(tmp, "ext")
    os.makedirs(ext_dir)
    ext = os.path.join(ext_dir, "model.onnx")
    onnx.save(onnx.load(path), ext, save_as_external_data=True,
              location="model.onnx.data", size_threshold=0)
    try:
        OpenVinoSession(ext)
        check("sidecar/present-loads", True)
    except Exception as e:
        check("sidecar/present-loads", False, "%s: %s" % (type(e).__name__, str(e)[:120]))
    os.remove(os.path.join(ext_dir, "model.onnx.data"))
    try:
        OpenVinoSession(ext)
        check("sidecar/missing-raises", False, "constructed a session over a graph whose weights are absent")
    except FileNotFoundError as e:
        check("sidecar/missing-raises", True)
        check("sidecar/names-the-file", "model.onnx.data" in str(e), str(e)[:160])
        check("sidecar/names-the-export", "export_fp32" in str(e), str(e)[:160])
    except Exception as e:
        check("sidecar/missing-raises", False, "wrong exception %s: %s" % (type(e).__name__, str(e)[:120]))

    # --- 6. unknown backend, before any file is read ------------------------------------------
    check("backends/listed", BACKENDS == ("onnxruntime", "openvino"), repr(BACKENDS))
    try:
        OnnxAgent(os.path.join(tmp, "does-not-exist"), backend="tensorrt")
        check("backends/unknown-raises", False)
    except ValueError as e:
        check("backends/unknown-raises", True)
        check("backends/unknown-names-valid-ones", "onnxruntime" in str(e) and "openvino" in str(e), str(e))
    except FileNotFoundError:
        check("backends/unknown-raises", False, "read the directory before validating the backend")

    # --- 7. concurrent calls on one agent ------------------------------------------------------
    states = ["billing state " * (i + 1) + "refund" for i in range(8)]
    serial = [a_ov.predict(s, QUESTIONS)["answers"] for s in states]
    got = [None] * len(states)
    errors = []

    def worker(i):
        try:
            for _ in range(5):
                got[i] = a_ov.predict(states[i], QUESTIONS)["answers"]
        except Exception as e:  # recorded, so a crash in a thread fails the test rather than vanishing
            errors.append("%s: %s" % (type(e).__name__, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(states))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("threads/no-errors", not errors, "; ".join(errors[:3]))
    check("threads/results-equal-serial", got == serial)

    # --- 8. an OpenVINO process never imports onnxruntime, torch or transformers ---------------
    probe = r"""
import sys
sys.path.insert(0, %r)
attempts = []
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("onnxruntime", "torch", "transformers"):
            attempts.append(name)
            raise RuntimeError("blocked: " + name)
sys.meta_path.insert(0, Blocker())
import laya_onnx
agent = laya_onnx.load(%r, backend="openvino")
agent.predict("billing refund state", {"n": {"type": "noul", "instructions": "refund state"}})
print("ATTEMPTS", attempts)
""" % (ROOT, ck)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    check("isolation/subprocess-ran", proc.returncode == 0, (proc.stderr or proc.stdout)[-400:])
    check("isolation/no-onnxruntime-torch-transformers", "ATTEMPTS []" in proc.stdout, proc.stdout[-200:])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all openvino backend tests passed")
sys.exit(1 if FAIL else 0)
