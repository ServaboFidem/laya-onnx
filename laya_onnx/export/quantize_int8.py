"""Dynamic int8 quantization of an fp32 laya-onnx export.

"Dynamic" here means the *weights* are quantized once, offline, to int8, while the *activation*
ranges are computed per-run at inference time. That is the right trade for this model: static
(calibrated) quantization would need a representative activation calibration set and would bake
in ranges that a multilingual encoder -- whose activation statistics shift with script and
language -- has no single representative sample for. Dynamic quantization needs no calibration
data at all and cannot be silently mis-calibrated by an unrepresentative one.

What it *can* do is move the logits, which is why this module is only half the story. Int8
weights perturb the pre-softmax logits by a small amount that argmax usually absorbs and
calibration does not: the label stays the same while the published probability drifts. The
measurement that decides whether that drift is acceptable lives in `laya_onnx/bench/eval_ece.py`,
and the temperature refit that can repair it lives in `laya_onnx/export/refit_temps.py`. Running
this module without running those is exactly the failure mode laya exists to avoid -- a model
that picks the right label and lies about how sure it is.

Three mechanical points about *this* graph, each of which costs a confusing failure if ignored:

1. **The fp32 export is a two-file artifact.** `export_fp32` writes `model.onnx` (~2.8 MB of
   graph) plus `model.onnx.data` (~1.29 GB of weights): torch's exporter externalizes the
   initializers, and the runtime requires both files to sit side by side. `quantize_dynamic` takes a *path*, and
   onnx resolves `.onnx.data` relative to that path's directory, so the two files must still be
   together when this runs. `_require_external_data` checks that up front rather than letting
   onnx fail later with a bare missing-file error that never mentions weights.

2. **The exporter's initializer `value_info` has to be stripped first.** `quantize_dynamic`
   begins by running `onnx.shape_inference.infer_shapes_path` over the model, and on this graph
   that raises `InferenceError: [ShapeInferenceError] Inferred shape and existing shape differ
   in dimension 0: (772) vs (256)` before a single weight is touched. Three measured facts pin
   it down:

     - `onnx.shape_inference.infer_shapes` on the *same* graph loaded with
       `load_external_data=False` succeeds, so the failure only appears once the initializer
       bytes are readable -- which is exactly the difference between the in-memory and the
       path-based entry point;
     - the graph carries 1617 `value_info` entries, 203 of which duplicate an initializer, and
       dropping just those 203 makes `infer_shapes_path` pass;
     - none of those 203 declares a shape that actually differs from its initializer's `dims`,
       so this is not a graph that lies about its own weights. (772 and 256 are the two
       dimensions of `model.act_head.0.weight`, a Gemm operand.)

   `value_info` is optional metadata for intermediate tensors; an entry for a tensor that is
   already an initializer carries no information the initializer does not, and ONNX Runtime
   re-infers shapes at session creation regardless. So `_strip_initializer_value_info` drops
   precisely those entries into a temporary graph-only copy and quantizes that. The copy is
   written *next to the original* because external-data locations resolve relative to the model
   file's own directory -- put the stripped graph anywhere else and its 1.29 GB of weights
   become unreachable. It is ~2.8 MB and is removed again in a `finally`.

3. **`MatMulConstBOnly` is not an optimization knob.** Without it, ONNX Runtime's dynamic
   quantizer will also rewrite MatMuls whose *both* inputs are activations -- which in a
   transformer means the attention `Q@K^T` and `attn@V` products. Those have no constant
   operand to pre-quantize, so each one gains a dynamic quantize/dequantize pair around
   activations whose range varies per token, buying numerical damage in exchange for no weight
   compression at all. Restricting the rewrite to MatMuls with a constant B keeps it to the
   linear layers, which is where the 1.29 GB actually lives.

Only the weights shrink; the graph, the config and the tokenizer are copied through unchanged,
so an int8 export directory loads with the same `laya_onnx.load()` as the fp32 one.
"""
import argparse
import os
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

_MODEL = "model.onnx"
_CONFIG = "rl_agent_config.json"
_TOKENIZER = "tokenizer"


def _require_external_data(onnx_path: str) -> None:
    """Fail early, and by name, if the sibling weight file is missing.

    onnx's own error for this case is raised from deep inside tensor loading and names only the
    absent file, with nothing to say that it is the model's weights or where it should have come
    from. Since the whole point of the fp32 export is that the two files travel together, an
    explicit check here is the difference between "copy model.onnx.data too" and a bug hunt.
    """
    data = onnx_path + ".data"
    if not os.path.exists(data):
        raise FileNotFoundError(
            "%r has no sibling weight file %r. The fp32 export is a two-file artifact "
            "(model.onnx carries the graph, model.onnx.data carries ~1.29 GB of weights) and "
            "onnx resolves the data file relative to the model path -- both must be present in "
            "the same directory before quantizing." % (onnx_path, os.path.basename(data))
        )


_STAGED_SUFFIX = ".quantize-staging.onnx"


def _strip_initializer_value_info(onnx_path: str) -> str:
    """Write a graph-only copy of `onnx_path` with initializer-duplicating `value_info` removed.

    Returns the path of the copy, which sits in the same directory as the original so that its
    external-data references (which onnx resolves relative to the model file) still point at the
    untouched `model.onnx.data`. The caller must delete it.

    The copy is loaded with `load_external_data=False` and saved without it, so this costs one
    ~2.8 MB write and never materialises the 1.29 GB of weights in memory. See point 2 of the
    module docstring for why the entries are dropped at all.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    initializers = {t.name for t in model.graph.initializer}
    keep = [vi for vi in model.graph.value_info if vi.name not in initializers]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)

    staged = os.path.splitext(onnx_path)[0] + _STAGED_SUFFIX
    onnx.save(model, staged)
    return staged


def quantize(fp32_path: str, int8_path: str) -> str:
    """Write an int8 copy of an fp32 export. Returns `int8_path`.

    Both arguments may be either a `model.onnx` file or a whole export *directory*. The
    directory form is the useful one -- `laya_onnx.load()` needs `rl_agent_config.json` and
    `tokenizer/` next to the graph, so quantizing the bare `.onnx` in isolation produces
    something no agent can load, and both the ECE measurement and the temperature refit that
    must follow this step work on loadable agents. When `fp32_path` is a directory, the config
    and tokenizer are copied into `int8_path` alongside the quantized graph.
    """
    as_dir = os.path.isdir(fp32_path)
    src_model = os.path.join(fp32_path, _MODEL) if as_dir else fp32_path
    dst_model = os.path.join(int8_path, _MODEL) if as_dir else int8_path

    if not os.path.exists(src_model):
        raise FileNotFoundError("no ONNX graph at %r" % src_model)
    _require_external_data(src_model)

    os.makedirs(os.path.dirname(os.path.abspath(dst_model)), exist_ok=True)

    # See point 2 of the module docstring: quantize_dynamic's first act is a path-based shape
    # inference pass that this graph fails, so it is handed a stripped graph-only copy instead.
    staged = _strip_initializer_value_info(src_model)
    try:
        quantize_dynamic(
            staged,
            dst_model,
            weight_type=QuantType.QInt8,
            # See the module docstring: this confines the rewrite to MatMuls with a constant B
            # (the linear layers), leaving the attention activation-by-activation products in
            # fp32.
            extra_options={"MatMulConstBOnly": True},
        )
    finally:
        if os.path.exists(staged):
            os.remove(staged)

    if as_dir:
        # The quantized graph is useless on its own: OnnxAgent.__init__ raises unless
        # rl_agent_config.json sits next to it (it carries max_len/head_max_len, which this
        # runtime refuses to guess), and TokenizerAdapter needs tokenizer/.
        src_cfg = os.path.join(fp32_path, _CONFIG)
        if not os.path.exists(src_cfg):
            raise FileNotFoundError(
                "%r is an export directory but has no %s; the int8 copy would not be "
                "loadable." % (fp32_path, _CONFIG)
            )
        shutil.copy2(src_cfg, os.path.join(int8_path, _CONFIG))
        src_tok = os.path.join(fp32_path, _TOKENIZER)
        if os.path.isdir(src_tok):
            shutil.copytree(src_tok, os.path.join(int8_path, _TOKENIZER), dirs_exist_ok=True)

    return int8_path


def _size_on_disk(path: str) -> int:
    """Total bytes of a model, counting any external-data sibling. A quantized model that keeps
    its weights external would otherwise look like a 3 MB file next to a 1.3 GB one."""
    if os.path.isdir(path):
        return sum(_size_on_disk(os.path.join(path, n)) for n in os.listdir(path))
    return os.path.getsize(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fp32", help="fp32 export directory (or model.onnx path)")
    ap.add_argument("int8", help="destination directory (or model.onnx path)")
    a = ap.parse_args(argv)
    out = quantize(a.fp32, a.int8)
    before, after = _size_on_disk(a.fp32), _size_on_disk(out)
    print("fp32 %.1f MB -> int8 %.1f MB (%.2fx)"
          % (before / 1e6, after / 1e6, before / max(after, 1)))
    print("Quantizing is not the end of this job: measure ECE against the int8 graph "
          "(laya_onnx/bench/eval_ece.py) before shipping it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
