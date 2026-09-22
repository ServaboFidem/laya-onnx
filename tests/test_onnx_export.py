"""Export a tiny DecisionModel and check the ONNX graph matches torch."""
import inspect
import os
import shutil
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import onnx                                                 # noqa: E402
import onnxruntime as ort                                   # noqa: E402
import torch                                                # noqa: E402
from transformers import AutoConfig, AutoModel              # noqa: E402

from laya.common import DecisionModel                       # noqa: E402
from laya_onnx.export.export_fp32 import (  # noqa: E402
    _build_parser, _copy_sidecars, _external_data_files, _verify_opset, export_fp32,
)
from laya_onnx.session import OnnxSession, declared_external_data   # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def tiny_model():
    cfg = AutoConfig.for_model("bert", hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=128, vocab_size=256, max_position_embeddings=128)
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    m = DecisionModel(enc, head_layers=1, n_act=2)
    m.eval()
    return m


def sample(batch=2, seq=32, markers=3, vocab=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, seq), generator=g)
    att = torch.ones((batch, seq), dtype=torch.long)
    att[-1, seq // 2:] = 0                     # ragged batch: padding must be exercised
    mpos = torch.randint(1, seq // 2, (batch, markers), generator=g)
    mmask = torch.ones((batch, markers), dtype=torch.bool)
    mmask[-1, -1] = False                      # ragged markers too
    qtype = torch.zeros((batch,), dtype=torch.long)
    return ids, att, mpos, mmask, qtype


tmp = tempfile.mkdtemp()
try:
    model = tiny_model()
    # opset=17 is passed EXPLICITLY. It is no longer the default (18 is), because 17 is
    # unreachable for mmBERT -- see export_fp32's docstring. This tiny BERT is small enough
    # that the down-converter succeeds on it, which is exactly why it stays here: it is the
    # only place the down-conversion path gets exercised at all.
    path = export_fp32(model, os.path.join(tmp, "model.onnx"), opset=17)
    check("export/file-exists", os.path.exists(path))

    # The written artifact must really be opset 17, not merely have been asked for it. torch
    # captures at 18 and down-converts through a best-effort version converter that is allowed
    # to give up and leave the file at 18 without raising.
    declared = [o.version for o in onnx.load(path).opset_import if o.domain in ("", "ai.onnx")]
    check("export/opset-is-17", declared == [17], "got %s" % declared)

    # The default must be 18. A default of 17 is a trap: on the checkpoint this port actually
    # ships it produces a graph that declares 17, fails to validate against 17, and is
    # rejected by onnxruntime -- i.e. it can only ever trip our own guard.
    check("export/default-opset-is-18",
          inspect.signature(export_fp32).parameters["opset"].default == 18,
          "got %r" % inspect.signature(export_fp32).parameters["opset"].default)
    check("export/cli-default-opset-is-18", _build_parser().get_default("opset") == 18,
          "got %r" % _build_parser().get_default("opset"))

    # ...and the guard that enforces it has to actually fire, or it is decoration.
    try:
        _verify_opset(path, 18)
        check("export/opset-guard-raises", False, "accepted a 17 artifact as opset 18")
    except RuntimeError:
        check("export/opset-guard-raises", True)

    ids, att, mpos, mmask, qtype = sample()
    with torch.no_grad():
        t_logits, t_act = model(ids, att, mpos, mmask, qtype)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o_logits, o_act = sess.run(
        ["logits", "act_logits"],
        {"input_ids": ids.numpy(), "attention_mask": att.numpy(), "marker_pos": mpos.numpy(),
         "marker_mask": mmask.numpy(), "qtype": qtype.numpy()},
    )

    d_logits = float(np.abs(t_logits.numpy() - o_logits).max())
    d_act = float(np.abs(t_act.numpy() - o_act).max())
    check("parity/logits<1e-4", d_logits < 1e-4, "max|delta|=%.3g" % d_logits)
    check("parity/act<1e-4", d_act < 1e-4, "max|delta|=%.3g" % d_act)

    # Dynamic axes: a different batch, sequence length AND marker count must all run.
    ids2, att2, mpos2, mmask2, _ = sample(batch=3, seq=48, markers=5, seed=1)
    qtype2 = torch.tensor([0, 1, 2])
    with torch.no_grad():
        t2, _ = model(ids2, att2, mpos2, mmask2, qtype2)
    o2, _ = sess.run(["logits", "act_logits"],
                     {"input_ids": ids2.numpy(), "attention_mask": att2.numpy(),
                      "marker_pos": mpos2.numpy(), "marker_mask": mmask2.numpy(),
                      "qtype": qtype2.numpy()})
    check("dynamic/shape", o2.shape == (3, 5), "got %s" % (o2.shape,))
    d2 = float(np.abs(t2.numpy() - o2).max())
    check("dynamic/parity<1e-4", d2 < 1e-4, "max|delta|=%.3g" % d2)

    # Masked-out markers must stay at the -1e4 floor, not leak a real score.
    check("masked-markers-floored", float(o_logits[-1, -1]) < -1e3, "got %.4g" % o_logits[-1, -1])

    # Task 6: the session wrapper must agree with raw onnxruntime.
    s_logits, s_act = OnnxSession(path).run(
        {"input_ids": ids.numpy(), "attention_mask": att.numpy(), "marker_pos": mpos.numpy(),
         "marker_mask": mmask.numpy(), "qtype": qtype.numpy()})
    check("session/logits-match", float(np.abs(s_logits - o_logits).max()) < 1e-6)
    check("session/act-match", float(np.abs(s_act - o_act).max()) < 1e-6)
    check("session/float32", str(s_logits.dtype) == "float32", str(s_logits.dtype))

    # Sidecars are required, not best-effort: an export directory without tokenizer/ is not
    # self-contained and TokenizerAdapter would find nothing at serving time. No weights needed
    # to test this - _copy_sidecars only ever looks at the filesystem.
    ck, out = os.path.join(tmp, "ckpt"), os.path.join(tmp, "out")
    os.makedirs(ck)
    os.makedirs(out)
    with open(os.path.join(ck, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    try:
        _copy_sidecars(ck, out)
        check("sidecars/missing-tokenizer-raises", False, "silently accepted a tokenizer-less checkpoint")
    except FileNotFoundError as e:
        check("sidecars/missing-tokenizer-raises", "tokenizer" in str(e), str(e)[:80])
    os.makedirs(os.path.join(ck, "tokenizer"))
    with open(os.path.join(ck, "tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    _copy_sidecars(ck, out)
    check("sidecars/copied", os.path.exists(os.path.join(out, "rl_agent_config.json"))
          and os.path.exists(os.path.join(out, "tokenizer", "tokenizer.json")))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------ _verify_opset branches
# The two guards added after the real mmBERT export are tested here against hand-built
# graphs, not against a real export. That is deliberate and necessary: the tiny BERT above
# takes the same C-API fallback path and happens to down-convert *successfully*, so no export
# this suite can afford structurally reaches either branch. Without these, the guard that
# caught the opset-17 disaster would itself be untested.

def _tiny_proto(opset, nodes, inputs, outputs, initializers=()):
    graph = onnx.helper.make_graph(list(nodes), "g", list(inputs), list(outputs),
                                   initializer=list(initializers))
    m = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset)])
    m.ir_version = 9
    return m


tmp_guard = tempfile.mkdtemp()
try:
    # (a) the exact shape the real failure had: a header claiming opset 17 over a node using
    # `num_outputs`, an attribute Split only gained in opset 18. The old guard read the header
    # and passed this; onnxruntime rejected it as INVALID_GRAPH at session creation.
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4])
    y1 = onnx.helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [2])
    y2 = onnx.helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [2])
    split = onnx.helper.make_node("Split", ["X"], ["Y1", "Y2"], name="node_Split_31",
                                  num_outputs=2)
    bad = os.path.join(tmp_guard, "bad_split.onnx")
    onnx.save(_tiny_proto(17, [split], [x], [y1, y2]), bad)

    declared_bad = [o.version for o in onnx.load(bad).opset_import if o.domain in ("", "ai.onnx")]
    check("guard/fixture-really-declares-17", declared_bad == [17], "got %s" % declared_bad)
    try:
        _verify_opset(bad, 17)
        check("guard/invalid-node-raises", False,
              "accepted a graph declaring 17 that does not validate against 17")
    except RuntimeError as e:
        check("guard/invalid-node-raises", True)
        # The message has to name what is wrong, or it sends the reader back to onnxruntime.
        check("guard/invalid-node-message-names-attribute", "num_outputs" in str(e), str(e)[:120])
        check("guard/invalid-node-message-names-node", "Split" in str(e), str(e)[:120])

    # (b) a graph at an opset it genuinely satisfies must pass -- otherwise (a) proves nothing
    # beyond "the guard raises on everything".
    relu = onnx.helper.make_node("Relu", ["X"], ["Z"], name="node_Relu_0")
    z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [4])
    good = os.path.join(tmp_guard, "good.onnx")
    onnx.save(_tiny_proto(17, [relu], [x], [z]), good)
    try:
        check("guard/valid-graph-passes", _verify_opset(good, 17) == 17)
    except Exception as e:
        check("guard/valid-graph-passes", False, "%s: %s" % (type(e).__name__, e))

    # (c) missing external data. This must be diagnosed as a missing sidecar, and must be
    # diagnosed BEFORE onnx.checker runs -- check_model resolves external data itself, so if
    # the checker went first the failure would arrive as a non-ValidationError that escapes
    # the RuntimeError handler entirely and reaches the caller as a raw onnx error.
    w = onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [4], vals=[0.0, 0.0, 0.0, 0.0])
    # Strip the inlined bytes and point the tensor at a sibling file instead, which is the
    # shape torch writes for any export past protobuf's 2 GB ceiling.
    w.ClearField("float_data")
    w.data_location = onnx.TensorProto.EXTERNAL
    w.external_data.add(key="location", value="model.onnx.data")
    w.external_data.add(key="offset", value="0")
    w.external_data.add(key="length", value="16")
    add = onnx.helper.make_node("Add", ["X", "W"], ["Z"], name="node_Add_0")
    ext = os.path.join(tmp_guard, "ext.onnx")
    onnx.save(_tiny_proto(17, [add], [x], [z], initializers=[w]), ext)

    found = _external_data_files(onnx.load(ext, load_external_data=False), ext)
    check("guard/external-data-discovered",
          found == [os.path.join(tmp_guard, "model.onnx.data")], "got %s" % found)
    try:
        _verify_opset(ext, 17)
        check("guard/missing-external-data-raises", False, "accepted a graph with absent weights")
    except FileNotFoundError as e:
        check("guard/missing-external-data-raises", True)
        check("guard/missing-external-data-names-the-file", "model.onnx.data" in str(e), str(e)[:140])
    except Exception as e:
        # Anything other than FileNotFoundError means the checker got there first, which is
        # the ordering bug this case exists to pin down.
        check("guard/missing-external-data-raises", False,
              "wrong exception type %s: %s" % (type(e).__name__, str(e)[:100]))

    # (d) a graph with no external data at all reports none -- the summary line in main()
    # iterates this, and a false positive there would print a file that does not exist.
    check("guard/no-external-data-reports-none",
          _external_data_files(onnx.load(good, load_external_data=False), good) == [])

    # (e) the *runtime* side of the same hazard. The exporter refused to write an unpaired
    # artifact; until now nothing refused to load one, so a deployment that copied only
    # model.onnx got onnxruntime's external-data error out of tensor loading, which says
    # nothing about the export. `laya_onnx.session` cannot use onnx to find the declared
    # sidecars (the serving extra ships neither onnx nor torch), so it walks the protobuf
    # itself -- and the first check below is that its answer is the same one onnx gives.
    check("session/walker-agrees-with-onnx-on-external-data",
          declared_external_data(ext) == _external_data_files(
              onnx.load(ext, load_external_data=False), ext),
          "%s vs %s" % (declared_external_data(ext),
                        _external_data_files(onnx.load(ext, load_external_data=False), ext)))
    check("session/walker-agrees-with-onnx-on-single-file",
          declared_external_data(good) == [], "got %s" % declared_external_data(good))
    try:
        OnnxSession(ext)
        check("session/missing-external-data-raises", False,
              "constructed a session over a graph whose weights are absent")
    except FileNotFoundError as e:
        check("session/missing-external-data-raises", True)
        check("session/missing-external-data-names-the-file", "model.onnx.data" in str(e),
              str(e)[:160])
        check("session/missing-external-data-names-the-export",
              "export_fp32" in str(e), str(e)[:160])
    except Exception as e:
        # An onnxruntime Fail here means the check did not run first, which is the whole bug.
        check("session/missing-external-data-raises", False,
              "wrong exception type %s: %s" % (type(e).__name__, str(e)[:120]))

    # And the guard must not become a second gate: a graph with no external data at all still
    # loads. Without this, "it raises" would be indistinguishable from "it always raises".
    try:
        OnnxSession(good)
        check("session/single-file-graph-still-loads", True)
    except Exception as e:
        check("session/single-file-graph-still-loads", False,
              "%s: %s" % (type(e).__name__, str(e)[:120]))
finally:
    shutil.rmtree(tmp_guard, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx export tests passed")
sys.exit(1 if FAIL else 0)
