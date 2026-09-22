"""OnnxAgent end-to-end: config-driven token budget, the k<2 refusal, and the answer shape.

This is the one ONNX test file allowed to import torch -- like test_onnx_export.py, it needs
torch to build and export a tiny DecisionModel as a fixture. `laya_onnx` itself must still never
import torch; that boundary is tests/test_onnx_no_torch.py's job, not this file's.

The central risk this file exists to catch (see the port's Task 7 brief): `build_sequence`
defaults to `max_len=512, head_max_len=192` -- upstream's English-checkpoint numbers, kept for
byte-parity -- and OnnxAgent must never fall back to them. If it silently did, every call would
run at a 512-token budget instead of the checkpoint's real one, quietly truncating exactly the
long states this port exists to handle, with no error anywhere. So the fixture config below uses
budget values that are deliberately far from both the defaults AND each other (777/333), and the
test monkeypatches `build_sequence` at its call site in `laya_onnx.runtime` to record the exact
`max_len`/`head_max_len` values it was invoked with -- proving those values are the config's, not
assumed from a passing pipeline run.
"""
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer                              # noqa: E402
from tokenizers.models import WordLevel                       # noqa: E402
from tokenizers.pre_tokenizers import Whitespace               # noqa: E402
from transformers import AutoConfig, AutoModel                # noqa: E402

from laya.common import DecisionModel                          # noqa: E402
from laya_onnx.export.export_fp32 import export_fp32           # noqa: E402
import laya_onnx.runtime as runtime                            # noqa: E402
from laya_onnx.runtime import OnnxAgent                        # noqa: E402
from laya_onnx.sequence import build_sequence as real_build_sequence  # noqa: E402

PASS, FAIL = [], []


_UNSET = object()


def check(name, got, want=_UNSET):
    """Two forms: `check(name, got, want)` compares by equality (matching this repo's other
    test_onnx_*.py files); `check(name, ok)` treats `got` as already boolean. Kept as one
    function, rather than two, so a call is never ambiguous about which form it used -- passing
    only `got` for something that was meant to be an equality check is exactly the bug class
    this docstring exists to prevent."""
    ok = got if want is _UNSET else (got == want)
    if ok:
        PASS.append(name)
    else:
        detail = "" if want is _UNSET else ("got %r, want %r" % (got, want))
        FAIL.append("%s %s" % (name, detail))


# Deliberately far from both build_sequence's defaults (512, 192) and each other, so a stray
# fallback to either the defaults or a swapped pair of arguments is caught, not coincidentally
# masked by a value that happens to match something else in the pipeline.
FIXTURE_MAX_LEN = 777
FIXTURE_HEAD_MAX_LEN = 333


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


def make_checkpoint(root, with_temperature=False):
    """A self-contained laya-onnx checkpoint dir: model.onnx + rl_agent_config.json + tokenizer/."""
    make_tokenizer(os.path.join(root, "tokenizer"))

    cfg = {"max_len": FIXTURE_MAX_LEN, "head_max_len": FIXTURE_HEAD_MAX_LEN}
    if with_temperature:
        cfg["temperature"] = [1.2, 0.9, 1.1]
        cfg["temperature_by_options"] = {"choice:2": 1.0}
    # else: leave temperature keys entirely absent, mirroring laya-multilingual's real shipped
    # config, which has no fitted temperatures at all (see AGENTS.md).
    with open(os.path.join(root, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    cfg2 = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=64, vocab_size=64, max_position_embeddings=128)
    enc = AutoModel.from_config(cfg2, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=1, n_act=2)
    model.eval()
    export_fp32(model, os.path.join(root, "model.onnx"))
    return root


CHOICE_Q = {"type": "choice", "instructions": "billing state",
            "criteria": {"good": "good state", "bad": "bad state", "ok": "ok state"}}
ONE_OPTION_Q = {"type": "choice", "instructions": "billing state", "criteria": {"good": "good state"}}


# --- config-driven budget: values reaching build_sequence come from the config -------------
tmp = tempfile.mkdtemp()
try:
    ck = make_checkpoint(tmp)

    captured = []
    orig = runtime.build_sequence

    def spy(tok, state, q, max_len, head_max_len, *a, **kw):
        captured.append((max_len, head_max_len))
        return real_build_sequence(tok, state, q, max_len, head_max_len, *a, **kw)

    runtime.build_sequence = spy
    try:
        agent = OnnxAgent(ck)
        check("budget/max_len-from-config", agent.cfg.get("max_len"), FIXTURE_MAX_LEN)
        check("budget/head_max_len-from-config", agent.cfg.get("head_max_len"), FIXTURE_HEAD_MAX_LEN)

        result = agent.system_one("billing refund charge state", {"q1": CHOICE_Q})
        check("budget/build_sequence-called", len(captured), 1)
        got_max_len, got_head_max_len = captured[0]
        check("budget/reached-build_sequence-max_len", got_max_len, FIXTURE_MAX_LEN)
        check("budget/reached-build_sequence-head_max_len", got_head_max_len, FIXTURE_HEAD_MAX_LEN)
        # And, explicitly, not upstream's byte-parity defaults -- the failure mode the brief
        # calls out by name: everything "succeeding" at 512/192 instead.
        check("budget/not-default-512", got_max_len != 512)
        check("budget/not-default-192", got_head_max_len != 192)
    finally:
        runtime.build_sequence = orig

    # --- result shape: {"model", "answers", "usage"} with the truncation addition -------------
    check("shape/model", result.get("model"), "laya-onnx")
    check("shape/has-answers", "answers" in result)
    check("shape/has-usage", "usage" in result)
    check("shape/answer-for-q1", "q1" in result["answers"])
    check("shape/usage-input-tokens-is-int", isinstance(result["usage"]["input_tokens"], int))
    check("shape/usage-output-tokens-zero", result["usage"]["output_tokens"], 0)
    check("shape/usage-has-truncated", "truncated" in result["usage"])
    check("shape/truncated-has-q1", "q1" in result["usage"]["truncated"])
    trep = result["usage"]["truncated"]["q1"]
    check("shape/truncated-report-keys",
          set(trep.keys()) >= {"state_tokens", "state_tokens_used", "dropped", "truncated"})

    # --- no fitted temperatures at all: the normal path for laya-multilingual ------------------
    check("temperature/absent-config-falls-back-default", agent.temperature, [1.0, 1.0, 1.0])
    check("temperature/absent-config-falls-back-empty-map", agent.temperature_by_options, {})

    # --- predict is system_one, and actually callable end to end -------------------------------
    check("predict/is-system_one", OnnxAgent.predict is OnnxAgent.system_one)
    result2 = agent.predict("billing refund charge state", {"q1": CHOICE_Q})
    check("predict/callable", "answers" in result2 and "q1" in result2["answers"])

    # --- k < 2 rejected before it ever reaches the ONNX graph -----------------------------------
    try:
        agent.system_one("billing state", {"q1": ONE_OPTION_Q})
        check("k<2/rejected", False)
    except ValueError as e:
        msg = str(e)
        check("k<2/rejected", True)
        check("k<2/message-names-question", "q1" in msg)
        check("k<2/message-explains-minimum", "2" in msg)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# --- a checkpoint WITH fitted temperatures still loads and uses them -----------------------
tmp2 = tempfile.mkdtemp()
try:
    ck2 = make_checkpoint(tmp2, with_temperature=True)
    agent2 = OnnxAgent(ck2)
    check("temperature/present-config-used", agent2.temperature, [1.2, 0.9, 1.1])
    check("temperature/present-config-bucket-used", agent2.temperature_by_options, {"choice:2": 1.0})
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

# --- missing rl_agent_config.json is a clear error, not an AttributeError later ------------
tmp3 = tempfile.mkdtemp()
try:
    make_tokenizer(os.path.join(tmp3, "tokenizer"))
    try:
        OnnxAgent(tmp3)
        check("config/missing-raises", False)
    except FileNotFoundError as e:
        check("config/missing-raises", True)
        check("config/missing-message-names-file", "rl_agent_config.json" in str(e))
finally:
    shutil.rmtree(tmp3, ignore_errors=True)


# --- a config present but missing max_len/head_max_len must raise, not fall back -----------
# This is the case the Important-1 review finding named directly: __init__ must never default
# either value (e.g. toward 1024/256) when the key is simply absent from an otherwise-valid
# config, because that "helpful" fallback is exactly what would silently build a 1024-token
# sequence against a graph an english-checkpoint export traced for 512. No tokenizer or
# model.onnx is needed for this check -- __init__ must raise before it ever gets to
# TokenizerAdapter or OnnxSession, so none is written here.
for missing_key, cfg in [
    ("max_len", {"head_max_len": FIXTURE_HEAD_MAX_LEN}),
    ("head_max_len", {"max_len": FIXTURE_MAX_LEN}),
]:
    tmp4 = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp4, "rl_agent_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        try:
            OnnxAgent(tmp4)
            check("config/missing-%s-raises" % missing_key, False)
        except ValueError as e:
            check("config/missing-%s-raises" % missing_key, True)
            check("config/missing-%s-message-names-key" % missing_key, missing_key in str(e))
    finally:
        shutil.rmtree(tmp4, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all onnx runtime tests passed")
sys.exit(1 if FAIL else 0)
