"""Export a tiny DecisionModel and check the ONNX graph matches torch."""
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
from laya_onnx.export.export_fp32 import _copy_sidecars, _verify_opset, export_fp32   # noqa: E402
from laya_onnx.session import OnnxSession                   # noqa: E402

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
    path = export_fp32(model, os.path.join(tmp, "model.onnx"))
    check("export/file-exists", os.path.exists(path))

    # The written artifact must really be opset 17, not merely have been asked for it. torch
    # captures at 18 and down-converts through a best-effort version converter that is allowed
    # to give up and leave the file at 18 without raising.
    declared = [o.version for o in onnx.load(path).opset_import if o.domain in ("", "ai.onnx")]
    check("export/opset-is-17", declared == [17], "got %s" % declared)

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

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx export tests passed")
sys.exit(1 if FAIL else 0)
