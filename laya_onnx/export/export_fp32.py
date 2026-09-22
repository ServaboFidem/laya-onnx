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

import onnx
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


def _external_data_files(model, path: str) -> list:
    """Absolute paths of the external-data files `model` references, first-seen order.

    `model` must have been loaded with `load_external_data=False`, so the initializers still
    carry their `external_data` entries rather than inlined bytes.

    Only top-level graph initializers are walked. That is not a general-purpose ONNX utility
    and is not meant to be: torch's exporter writes every weight as a top-level initializer of
    the main graph, and this function exists to check *this* exporter's own output, not to
    survive arbitrary graphs. A subgraph-nested external tensor would be missed, which would
    cost us the friendly error below and nothing else - the checker and onnxruntime still fail.
    """
    base = os.path.dirname(os.path.abspath(path))
    seen, out = set(), []
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for kv in tensor.external_data:
            if kv.key == "location" and kv.value not in seen:
                seen.add(kv.value)
                out.append(os.path.join(base, kv.value))
    return out


def _verify_opset(path: str, want: int) -> int:
    """Raise unless the file at `path` really declares opset `want` *and* validates against it.

    `opset_version=` is a request, not a guarantee. torch.export captures at its own opset (18
    at the time of writing) and then asks a version converter to walk the graph back down. That
    converter is best-effort, and torch says so in its own warning: "If version conversion is
    unsuccessful, the opset version of the exported model will be kept at 18." Nothing about a
    failed down-conversion is an error - `torch.onnx.export` still returns and still writes a
    file, so without this check the caller, the CLI's own summary line and the port plan would
    all go on believing the artifact is opset 17 while it is opset 18.

    Down-conversion succeeds on a small BERT. The encoders this port actually ships - ModernBERT
    and mmBERT - are exactly the op mix a version converter gives up on, and the failure would
    land in Task 8 as a runtime incompatibility on some other host rather than here as an export
    that refused to finish. So: read the artifact back and make the mismatch loud.
    """
    model = onnx.load(path, load_external_data=False)
    # The default ONNX domain is spelled "" (and, historically, "ai.onnx"); custom-op domains
    # carry their own independent versions and are not what `opset_version` controls.
    got = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    if not got:
        raise RuntimeError("exported %s declares no default-domain opset" % path)
    if len(got) > 1 or got[0] != want:
        raise RuntimeError(
            "exported %s declares opset %s, not the requested %d. torch.export captures at a "
            "newer opset and down-converts; that conversion is best-effort and evidently did "
            "not take here. Re-export with --opset %s, or fix the converter, rather than "
            "shipping an artifact whose opset nobody agrees on." % (path, got, want, got[0])
        )

    # Matching the declared opset is necessary but NOT sufficient, and the difference is not
    # theoretical: exporting the real multilingual checkpoint at opset 17 produced a file that
    # declared opset 17 and passed the check above, while containing
    #     Split(..., num_outputs=2)
    # - an attribute Split only gained in opset 18. The C API fallback converter had rewritten
    # the *header* to 17 and left that node untouched, so the artifact was internally
    # inconsistent: every downstream tool that trusts the header rejects it. onnxruntime's
    # rejection is the one that matters, and it arrives at *load* time on the serving host:
    #     INVALID_GRAPH : ... Unrecognized attribute: num_outputs for operator Split
    # onnx.checker validates each node against the schema for the declared opset and catches
    # exactly this, here, on the machine doing the export. It is preferred over an
    # onnxruntime load because the exporter must not acquire a runtime dependency just to
    # check itself, and because the checker names the offending node rather than the file.

    # External data first, and deliberately before the checker. An fp32 mmBERT export is
    # ~1.29 GB, far past protobuf's 2 GB message ceiling, so torch splits it: `model.onnx` is a
    # ~2.8 MB graph and `model.onnx.data` beside it holds 99.8% of the bytes. The two travel
    # together or not at all.
    #
    # This check has to run first because `onnx.checker.check_model(path)` *resolves* external
    # data itself. With the `.data` file absent it raises something that is not a
    # ValidationError, sails straight past the `except` below, and reaches the caller as a raw
    # onnx error about a missing file - which is precisely the unhelpful failure this is here
    # to replace. Checking first means a missing sidecar is diagnosed as a missing sidecar.
    for ext in _external_data_files(model, path):
        if not os.path.exists(ext):
            raise FileNotFoundError(
                "exported %s references external data %r, which is not there. The graph and "
                "its weights are two files and must be deployed together - copying model.onnx "
                "alone gives you a 2.8 MB graph with no weights, and onnxruntime will fail at "
                "session creation on the serving host rather than here."
                % (path, os.path.basename(ext))
            )

    # `check_model` is given the path, not the loaded proto: these graphs exceed the 2 GB
    # protobuf ceiling and carry their weights in a sibling `.data` file, which only the
    # path-taking form knows how to resolve -- and which the loop above has just confirmed
    # is actually there.
    try:
        onnx.checker.check_model(path, full_check=False)
    except onnx.checker.ValidationError as exc:
        raise RuntimeError(
            "exported %s declares opset %d but does not validate against it: %s. "
            "This is the down-converter having relabelled the graph without actually "
            "converting every node. The file loads nowhere -- onnxruntime rejects it as "
            "INVALID_GRAPH at session creation. Re-export at an opset the capture can reach "
            "natively (torch.export captures at 18) rather than shipping it."
            % (path, want, exc)
        ) from exc
    return got[0]


def export_fp32(model, out_path: str, opset: int = 18) -> str:
    """Trace `model` to ONNX at `out_path`, with batch, sequence and marker axes dynamic.

    Returns the path written, so a caller can chain on it.

    The default is 18 because 17 is unreachable for the encoders this port actually ships.
    torch.export captures at 18; asking for 17 hands the graph to a version converter that,
    on mmBERT, falls back to the onnx C API, rewrites the header to 17 and leaves a
    `Split(num_outputs=2)` node - an opset-18 attribute - in place. The result declares 17,
    does not validate against 17, and onnxruntime rejects it as INVALID_GRAPH. `_verify_opset`
    now catches that, but a default that can only ever trip our own guard is a trap, not a
    conservative choice. Pass `opset=17` explicitly if you want to exercise down-conversion on
    a model small enough for it to succeed.
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
    # `model.eval()` above does not put the wrapper itself in eval mode - `ExportWrapper` is a
    # fresh nn.Module whose own `.training` is True, and the exporter reports the mode of the
    # module it was handed. The captured graph is unaffected (the submodule that owns the
    # dropout is already in eval), but the exporter emits a training-mode warning that is
    # indistinguishable from the one you would get if dropout really were leaking into the
    # graph. Silencing it keeps that warning meaningful.
    wrapper = ExportWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper, args, out_path,
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

    _verify_opset(out_path, opset)
    return out_path


# Files the runtime needs alongside the graph. `rl_agent_config.json` carries the question-type
# temperatures and the token budget; `tokenizer/` is what `laya_onnx.tokenizer.TokenizerAdapter`
# loads. Copying them makes the output directory self-contained, so deploying the ONNX runtime
# never means also fetching the torch checkpoint it came from.
#
# Both are *required*, not best-effort. An earlier version skipped whatever was absent and
# reported "(nothing found)" while still exiting 0, which produced exactly the artifact this
# function exists to prevent: a graph with no tokenizer beside it, where `TokenizerAdapter`
# finds nothing at serving time and the failure surfaces on a production host instead of on the
# machine that did the export. A checkpoint missing either one is not a checkpoint this runtime
# can be built from, so say so here and stop.
_SIDECAR_FILES = ("rl_agent_config.json",)
_SIDECAR_DIRS = ("tokenizer",)


def _copy_sidecars(model_dir: str, out_dir: str) -> list:
    missing = [n for n in _SIDECAR_FILES if not os.path.exists(os.path.join(model_dir, n))]
    missing += [n + "/" for n in _SIDECAR_DIRS if not os.path.isdir(os.path.join(model_dir, n))]
    if missing:
        raise FileNotFoundError(
            "checkpoint %s is missing %s, which the ONNX runtime needs beside the graph. "
            "Export it from a complete laya checkpoint - an output directory without these is "
            "not self-contained and will fail at load time, not here."
            % (model_dir, ", ".join(missing))
        )

    copied = []
    for name in _SIDECAR_FILES:
        shutil.copy2(os.path.join(model_dir, name), os.path.join(out_dir, name))
        copied.append(name)
    for name in _SIDECAR_DIRS:
        shutil.copytree(os.path.join(model_dir, name), os.path.join(out_dir, name), dirs_exist_ok=True)
        copied.append(name + "/")
    return copied


def _build_parser():
    """The CLI's argument parser, built here rather than inline in `main()`.

    Extracted purely so a test can assert what the CLI defaults to without needing weights:
    `main()` parses and then immediately reaches for a checkpoint, so there is no way to
    observe its defaults by calling it. A test that rebuilt an equivalent parser of its own
    would assert nothing at all -- it would compare a hardcoded default against itself.
    """
    ap = argparse.ArgumentParser(description="Export a laya checkpoint to fp32 ONNX.")
    ap.add_argument("--model-dir", required=True,
                    help="directory holding rl_agent_config.json, model.safetensors and tokenizer/")
    ap.add_argument("--out", required=True, help="output directory for model.onnx and its sidecars")
    # Defaults to 18, matching export_fp32 -- see its docstring: 17 cannot be reached for
    # mmBERT and only produces an artifact this module's own guard rejects.
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset to write (default 18; 17 is unreachable for mmBERT)")
    return ap


def main(argv=None) -> int:
    # Import inside main() so the module stays importable (and `export_fp32` usable against a
    # model you already hold) without transformers or safetensors being present.
    from safetensors.torch import load_file

    from laya.common import build_model

    args = _build_parser().parse_args(argv)

    model_dir, out_dir = args.model_dir, args.out
    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    for p in (cfg_path, weights_path):
        if not os.path.exists(p):
            raise FileNotFoundError("not a laya checkpoint directory: missing %s" % p)
    # The sidecars are copied after the export, but they are checked here so an incomplete
    # checkpoint fails in a second rather than after several minutes of tracing a 421M encoder.
    if not os.path.isdir(os.path.join(model_dir, "tokenizer")):
        raise FileNotFoundError(
            "not a laya checkpoint directory: missing %s. The ONNX runtime loads its tokenizer "
            "from beside the graph, so an export without it cannot serve."
            % os.path.join(model_dir, "tokenizer")
        )

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
    # Name the weights file explicitly. It is 99.8% of the export and it is easy to miss:
    # `model.onnx` alone looks like a complete artifact at 2.8 MB, and a deploy that copies
    # only the three documented names ships a graph with no weights in it.
    for ext in _external_data_files(onnx.load(out_path, load_external_data=False), out_path):
        print("wrote %s (external weights -- deploy it beside model.onnx, it is not optional)"
              % ext)
    # `copied` is never empty: `_copy_sidecars` raises rather than skipping a missing sidecar.
    print("copied alongside: %s" % ", ".join(copied))
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
