"""Export DecisionModel to ONNX (fp32). One of only two places that may import torch.

This module runs once, offline, on a machine that has PyTorch, `onnx` and `onnxscript`
installed. The artifact it writes is what the runtime loads; nothing under `laya_onnx/` outside
this package imports torch, so the serving host needs none of those three.

Three things about the traced graph are load-bearing, and each of them fails silently rather
than loudly if you get it wrong - the export succeeds, the sample reproduces, and only a real
request exposes the damage. They are documented at their sites below:

  1. the traced signature must be exactly the five runtime tensors (see `ExportWrapper`);
  2. the sample must carry at least two markers (see `export_fp32`);
  3. batch, sequence *and* marker axes must all be dynamic (see `export_fp32`).
"""
import argparse
import json
import os
import shutil

import torch

# ModernBERT's encoder config defaults `reference_compile` to "auto", which routes the forward
# pass through torch.compile. The compiled path does not trace: the exporter either errors out
# or bakes in a graph built from compiled artifacts. Flipping it off is a no-op for encoders
# that have no such attribute, which is why this is a guarded setattr rather than an assumption.
_COMPILE_FLAG = "reference_compile"


def _disable_reference_compile(model) -> bool:
    """Turn off ModernBERT's compiled path. Returns whether the attribute was actually present."""
    cfg = getattr(getattr(model, "encoder", None), "config", None)
    if cfg is None or not hasattr(cfg, _COMPILE_FLAG):
        return False
    setattr(cfg, _COMPILE_FLAG, False)
    return True


class ExportWrapper(torch.nn.Module):
    """Pins the traced signature to the five runtime tensors.

    `DecisionModel.forward` also takes `detach_encoder`, which is training-only: it stops the
    gradient at the encoder boundary and has no meaning at inference. Leaving it in the traced
    signature invites a caller to pass it and get a graph that silently ignores it - the ONNX
    graph has no such input, so the value would be dropped on the floor with no error. Dropping
    it here makes the contract explicit: five tensors in, two tensors out.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def export_fp32(model, out_path: str, opset: int = 17) -> str:
    """Trace `model` to ONNX at `out_path`, with batch, sequence and marker axes dynamic.

    Returns the path written, so a caller can chain on it.
    """
    model.eval()
    _disable_reference_compile(model)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # At least 2 markers: laya/common.py:115 branches on `p.size(-1) >= 2` and only the true side
    # calls `p.topk(2, -1)`. A tracer resolves that Python `if` once, against the sample, and
    # writes whichever side it took into the graph permanently. Tracing with a single marker
    # would bake in the one-option padding branch, and every real multi-option question would
    # then get act-head features computed as if it had exactly one option. Three markers here
    # rather than two so the sample is not itself sitting on the boundary.
    batch, seq, markers = 2, 32, 3
    vocab = int(getattr(model.encoder.config, "vocab_size", 256))
    args = (
        torch.randint(1, vocab, (batch, seq)),
        torch.ones((batch, seq), dtype=torch.long),
        torch.randint(1, seq // 2, (batch, markers)),
        torch.ones((batch, markers), dtype=torch.bool),
        torch.zeros((batch,), dtype=torch.long),
    )

    # Three dynamic axes, not two. Batch and sequence are the obvious ones, but the marker count
    # is the number of options on the question being asked and varies request to request - a
    # graph with a frozen marker axis passes every test that reuses the export sample and fails
    # on the second real question. The declared input dtypes (int64 everywhere except the bool
    # marker_mask) are exactly what `laya_onnx.collate.collate` emits, so no cast is needed in
    # the hot path.
    with torch.no_grad():
        torch.onnx.export(
            ExportWrapper(model), args, out_path,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "markers"},
                "marker_mask": {0: "batch", 1: "markers"},
                "qtype": {0: "batch"},
                "logits": {0: "batch", 1: "markers"},
                "act_logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            # dynamo=True is not a preference, it is the only exporter that gets the sequence
            # axis right here. The legacy TorchScript exporter traces `nn.MultiheadAttention`
            # in the decision head through `F.multi_head_attention_forward`, whose
            # `q.view(tgt_len, bsz * num_heads, head_dim)` (torch/nn/functional.py, the line
            # reached on every path) reads `tgt_len` off a Python unpack of `query.shape`. The
            # tracer freezes that into a Constant, so the exported graph carries the sample's
            # sequence length inside the head's attention and dies on the first request of a
            # different length:
            #     Reshape ... Input shape:{48,3,64}, requested shape:{32,3,64}
            # Only `bsz` survives as a Shape/Gather. `do_constant_folding=False` does not help
            # - the constant is in the trace, not the folder. torch.export keeps all three
            # axes symbolic instead. This requires `onnxscript` at export time (never at
            # runtime); torch raises a clear ModuleNotFoundError if it is missing.
            #
            # verbose=False because the exporter's progress lines contain U+2705, which raises
            # UnicodeEncodeError on a Windows console left at cp1252.
            dynamo=True,
            verbose=False,
        )
    return out_path


# Files the runtime needs alongside the graph. `rl_agent_config.json` carries the question-type
# temperatures and the token budget; `tokenizer/` is what `laya_onnx.tokenizer.TokenizerAdapter`
# loads. Copying them makes the output directory self-contained, so deploying the ONNX runtime
# never means also fetching the torch checkpoint it came from.
_SIDECAR_FILES = ("rl_agent_config.json",)
_SIDECAR_DIRS = ("tokenizer",)


def _copy_sidecars(model_dir: str, out_dir: str) -> list:
    copied = []
    for name in _SIDECAR_FILES:
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            copied.append(name)
    for name in _SIDECAR_DIRS:
        src = os.path.join(model_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out_dir, name), dirs_exist_ok=True)
            copied.append(name + "/")
    return copied


def main(argv=None) -> int:
    # Import inside main() so the module stays importable (and `export_fp32` usable against a
    # model you already hold) without transformers or safetensors being present.
    from safetensors.torch import load_file

    from laya.common import build_model

    ap = argparse.ArgumentParser(description="Export a laya checkpoint to fp32 ONNX.")
    ap.add_argument("--model-dir", required=True,
                    help="directory holding rl_agent_config.json, model.safetensors and tokenizer/")
    ap.add_argument("--out", required=True, help="output directory for model.onnx and its sidecars")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    model_dir, out_dir = args.model_dir, args.out
    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    for p in (cfg_path, weights_path):
        if not os.path.exists(p):
            raise FileNotFoundError("not a laya checkpoint directory: missing %s" % p)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(weights_path), strict=True)
    had_compile_flag = _disable_reference_compile(model)

    os.makedirs(out_dir, exist_ok=True)
    out_path = export_fp32(model, os.path.join(out_dir, "model.onnx"), opset=args.opset)
    copied = _copy_sidecars(model_dir, out_dir)

    print("wrote %s (opset %d, reference_compile %s)"
          % (out_path, args.opset, "disabled" if had_compile_flag else "absent"))
    print("copied alongside: %s" % (", ".join(copied) if copied else "(nothing found)"))
    return 0


if __name__ == "__main__":
    # transformers probes for TensorFlow at import and its abseil runtime can deadlock model
    # construction. Every entry point in this repo sets these before the first transformers
    # import; main() reaches transformers through laya.common.build_model, so the guard has to
    # be in place before main() runs, not merely before the export call.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
