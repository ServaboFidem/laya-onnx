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

Four mechanical points about *this* graph, each of which costs a confusing failure if ignored:

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

4. **`quantize_embeddings` controls the token-embedding table, and neither setting rescues
   this checkpoint.** `quantize_dynamic`'s default `op_types_to_quantize` is every key of ONNX
   Runtime's `IntegerOpsRegistry`, which includes `Gather`. This graph's single `Gather` named
   `node_embedding` reads `model.encoder.embeddings.tok_embeddings.weight`, a [256000, 768]
   table holding ~196M of the checkpoint's ~322M parameters, and quantizing a `Gather` operand
   is *per-tensor*: the written graph carries a scalar `..._scale` and `..._zero_point`, one
   pair for all 256,000 rows.

   That looks like an obvious culprit for the accuracy damage int8 does here, and it was
   tested as one. **It is not the cause.** Both configurations were measured on the same 1500
   labelled examples and the same held-out split, with an identical body (100 `MatMulInteger`
   nodes either way -- the only difference between the two graphs is the embedding table):

     - `noul:2` held-out accuracy is **0.7800 either way**, against fp32's 0.8833.
     - Per suite, sparing the table made `toxic_chat` *worse* (0.5733 -> 0.5400) and `emotion`
       slightly better (0.5000 -> 0.5100); fp32-agreement on `toxic_chat` fell 0.7300 -> 0.6967.
     - Max absolute logit deviation from fp32 stayed in the same 6-19 nat range.

   So the damage lives in the body's 100 quantized linear layers, not in the embedding lookup.
   Calibration survives either way (after a temperature refit, ECE is within 0.02 of fp32 in
   every bucket); the labels do not. `laya_onnx/README.md` carries the full three-way table and
   the conclusion, which is to ship fp32.

   The default is nonetheless `quantize_embeddings=False`. Not because it recovers accuracy --
   it does not, and no comment here should be read as claiming otherwise -- but because a
   single scale spanning 196M parameters is indefensible on its face, the exclusion costs no
   body quantization at all, and the size it costs is moot for a configuration that is not
   recommended for shipping. It restricts the op set to `MatMul`, the only registry entry this
   graph contains besides `Gather` and `Transpose` (`Transpose` being a no-op unless its input
   is already quantized). Size: 1290.5 MB fp32 -> 324.7 MB quantizing everything -> 914.6 MB
   sparing the table.

Only the weights shrink; the graph, the config and the tokenizer are copied through unchanged,
so an int8 export directory loads with the same `laya_onnx.load()` as the fp32 one.
"""
import argparse
import os
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

_SUMMARY = "Dynamic int8 quantization of an fp32 laya-onnx export."

# The one registry op this graph is quantized through by default. The default set also contains
# `Gather`, and this graph's only `Gather` over a large initializer is the 256k-row
# token-embedding table, which `Gather` quantization would collapse onto a single scale.
# Excluding it was *measured* and did NOT recover the accuracy int8 costs here -- see point 4 of
# the module docstring for the numbers. `Transpose` is also in the default set but is a no-op
# unless its input is already quantized, so naming MatMul alone loses nothing measurable.
_BODY_OP_TYPES = ["MatMul"]

# The initializer that must stay fp32 when `quantize_embeddings` is False. Named explicitly so
# `assert_embeddings_not_quantized` can prove it from the *written* graph rather than trusting
# that the op-type restriction did what it was supposed to.
_EMBEDDING_INITIALIZER = "model.encoder.embeddings.tok_embeddings.weight"

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


def quantized_initializers(onnx_path: str) -> list:
    """`(name, onnx dtype name, dims)` for every initializer the quantizer rewrote.

    ONNX Runtime writes a quantized weight as a new initializer named `<original>_quantized`
    and drops the fp32 original, so the suffix is a reliable marker of what was actually
    touched. Reported by the CLI because "which tensors became int8" is the one question the
    size number cannot answer -- a 3.97x shrink looks like a success whether it came from the
    100 body MatMuls or from crushing a 256k-row embedding table into one scale.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    out = []
    for t in model.graph.initializer:
        if t.name.endswith("_quantized"):
            out.append((t.name, onnx.TensorProto.DataType.Name(t.data_type), list(t.dims)))
    return out


def assert_embeddings_not_quantized(onnx_path: str) -> None:
    """Raise unless the token-embedding table survived in fp32 in the *written* graph.

    This is deliberately a check on the artifact, not on the arguments that produced it. The
    op-type restriction is an instruction to a third-party quantizer; whether it was honoured
    is a property of the file. "We passed the right flag" and "the flag took effect" are
    different claims, and only the second one is checkable here -- which is the whole reason
    the exclusion experiment in point 4 could be trusted to have actually excluded anything.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    by_name = {t.name: t for t in model.graph.initializer}

    bad = by_name.get(_EMBEDDING_INITIALIZER + "_quantized")
    if bad is not None:
        raise RuntimeError(
            "%r quantized the token-embedding table: found initializer %r with dtype %s and "
            "shape %s. That is a per-tensor quantization of ~196M parameters across 256k rows "
            "sharing one scale. Measured on this checkpoint, excluding it does NOT recover the "
            "accuracy int8 costs -- the damage is elsewhere -- so this is not the error that "
            "explains a bad int8 model; it means the op-type restriction did not take effect "
            "and the graph is not the one that was asked for."
            % (onnx_path, bad.name, onnx.TensorProto.DataType.Name(bad.data_type),
               list(bad.dims)))

    kept = by_name.get(_EMBEDDING_INITIALIZER)
    if kept is None:
        raise RuntimeError(
            "%r has no initializer named %r at all, quantized or otherwise. This check knows "
            "the multilingual (mmBERT) export's tensor names; it cannot vouch for a graph it "
            "does not recognise, and silently passing would be worse than failing."
            % (onnx_path, _EMBEDDING_INITIALIZER))
    if kept.data_type != onnx.TensorProto.FLOAT:
        raise RuntimeError(
            "%r's %r is %s, not FLOAT."
            % (onnx_path, _EMBEDDING_INITIALIZER,
               onnx.TensorProto.DataType.Name(kept.data_type)))


def quantize(fp32_path: str, int8_path: str, quantize_embeddings: bool = False) -> str:
    """Write an int8 copy of an fp32 export. Returns `int8_path`.

    Both arguments may be either a `model.onnx` file or a whole export *directory*. The
    directory form is the useful one -- `laya_onnx.load()` needs `rl_agent_config.json` and
    `tokenizer/` next to the graph, so quantizing the bare `.onnx` in isolation produces
    something no agent can load, and both the ECE measurement and the temperature refit that
    must follow this step work on loadable agents. When `fp32_path` is a directory, the config
    and tokenizer are copied into `int8_path` alongside the quantized graph.

    `quantize_embeddings=False` (the default) restricts the rewrite to `MatMul`, leaving the
    256k-row token-embedding table in fp32. Passing True reproduces `quantize_dynamic`'s stock
    behaviour. **Both were measured and neither preserves this checkpoint's accuracy** -- the
    `noul:2` bucket lands on 0.7800 either way against fp32's 0.8833. See point 4 of the module
    docstring: the parameter exists so that comparison stays reproducible, and the default is
    the more conservative graph rather than a fix.
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
            # Point 4: restricting the op set is what spares the token-embedding table. Passing
            # None here means "every IntegerOpsRegistry key", which includes Gather.
            op_types_to_quantize=None if quantize_embeddings else list(_BODY_OP_TYPES),
            # See the module docstring: this confines the rewrite to MatMuls with a constant B
            # (the linear layers), leaving the attention activation-by-activation products in
            # fp32.
            extra_options={"MatMulConstBOnly": True},
        )
    finally:
        if os.path.exists(staged):
            os.remove(staged)

    # Prove it from the written graph rather than trusting the op-type restriction. An
    # op_types_to_quantize that silently stopped covering this case would produce a graph that
    # is not the one the caller asked for, and would do it silently -- the model still loads
    # and still answers. That mattered most while the exclusion was being used as an experiment:
    # a flag that quietly did nothing would have produced two identical graphs and a confident
    # null result.
    if not quantize_embeddings:
        assert_embeddings_not_quantized(dst_model)

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


def _numel(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _size_on_disk(path: str) -> int:
    """Total bytes of a model, counting any external-data sibling. A quantized model that keeps
    its weights external would otherwise look like a 3 MB file next to a 1.3 GB one."""
    if os.path.isdir(path):
        return sum(_size_on_disk(os.path.join(path, n)) for n in os.listdir(path))
    return os.path.getsize(path)


def main(argv=None) -> int:
    # `__doc__` is None under `python -OO`, which strips docstrings; `.splitlines()[0]` on it is
    # an AttributeError that only ever appears in an optimized run, i.e. the one place nobody
    # tests. _SUMMARY is the same sentence as a real constant.
    ap = argparse.ArgumentParser(description=_SUMMARY)
    ap.add_argument("fp32", help="fp32 export directory (or model.onnx path)")
    ap.add_argument("int8", help="destination directory (or model.onnx path)")
    ap.add_argument("--quantize-embeddings", action="store_true",
                    help="also quantize the token-embedding table (stock quantize_dynamic "
                         "behaviour). Smaller, and measured to be no worse for accuracy than "
                         "excluding it -- see point 4 of this module's docstring; neither "
                         "configuration is recommended for shipping.")
    a = ap.parse_args(argv)
    out = quantize(a.fp32, a.int8, quantize_embeddings=a.quantize_embeddings)
    before, after = _size_on_disk(a.fp32), _size_on_disk(out)
    print("fp32 %.1f MB -> int8 %.1f MB (%.2fx)"
          % (before / 1e6, after / 1e6, before / max(after, 1)))

    model_path = os.path.join(out, _MODEL) if os.path.isdir(out) else out
    quantized = quantized_initializers(model_path)
    print("%d initializers quantized; largest:" % len(quantized))
    for name, dtype, dims in sorted(quantized, key=lambda x: -_numel(x[2]))[:5]:
        print("   %-8s %-16s %s" % (dtype, dims, name))
    print("token-embedding table quantized: %s" % a.quantize_embeddings)

    print("Quantizing is not the end of this job: measure ECE against the int8 graph "
          "(laya_onnx/bench/eval_ece.py) before shipping it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
