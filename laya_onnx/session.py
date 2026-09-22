"""onnxruntime session wrapper. Runtime code: numpy and onnxruntime only, never torch.

Thread count is the caller's choice. onnxruntime defaults `intra_op_num_threads` to every
physical core, which oversubscribes badly when the host already runs several workers - each
worker then spawns a full-width thread pool and they spend their time descheduling each other.
Leaving `threads=None` keeps onnxruntime's default for the single-process case; passing a small
number is the right move under a process pool or a container with a CPU quota.

The fixed `_INPUTS` tuple is deliberate. `collate()` emits exactly these five arrays with
exactly the dtypes the exported graph declares (int64, except the bool `marker_mask`), so the
feed is built by selection rather than by passing the dict through: an extra key would be a
sign the two sides have drifted, and onnxruntime would reject it at a less obvious moment.
"""
import os

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
_OUTPUTS = ["logits", "act_logits"]

# Protobuf field numbers on the path ModelProto -> GraphProto -> TensorProto ->
# StringStringEntryProto, from onnx.proto3. Hard-coded because the runtime extra deliberately
# does not ship `onnx` (see pyproject) -- the serving process holds onnxruntime and numpy and
# nothing else, so the declared external-data locations are read with a 40-line varint walker
# rather than by loading the graph through the onnx Python package.
_F_MODEL_GRAPH = 7
_F_GRAPH_INITIALIZER = 5
_F_TENSOR_EXTERNAL_DATA = 13
_F_TENSOR_DATA_LOCATION = 14
_TENSOR_LOCATION_EXTERNAL = 1
_F_ENTRY_KEY = 1
_F_ENTRY_VALUE = 2


def _fields(buf: bytes):
    """Yield `(field_number, wire_type, payload)` for one protobuf message.

    `payload` is the raw bytes for wire type 2 and the decoded integer for wire types 0/1/5.
    Unknown fields are skipped by wire type, which is what makes this safe against every part
    of the schema it does not model.
    """
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
            yield field, wire, val
        elif wire == 2:
            ln, i = _varint(buf, i)
            if i + ln > n:
                raise ValueError("truncated length-delimited field")
            yield field, wire, buf[i:i + ln]
            i += ln
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError("unsupported wire type %d" % wire)


def _varint(buf: bytes, i: int) -> Tuple[int, int]:
    val, shift = 0, 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def declared_external_data(model_path: str) -> List[str]:
    """Absolute paths of the sidecar weight files `model_path`'s initializers declare.

    Empty for a single-file model, and empty for anything this walker cannot parse: the point
    of the check in `OnnxSession.__init__` is to turn one specific, known, high-cost deployment
    mistake into a clear message, never to become a second gate that can refuse a model
    onnxruntime would happily load.
    """
    try:
        with open(model_path, "rb") as f:
            blob = f.read()
    except OSError:
        return []

    base = os.path.dirname(os.path.abspath(model_path))
    out: List[str] = []
    seen = set()
    try:
        for field, wire, payload in _fields(blob):
            if field != _F_MODEL_GRAPH or wire != 2:
                continue
            for gfield, gwire, gpayload in _fields(payload):
                if gfield != _F_GRAPH_INITIALIZER or gwire != 2:
                    continue
                external, location = False, None
                for tfield, twire, tpayload in _fields(gpayload):
                    if tfield == _F_TENSOR_DATA_LOCATION and twire == 0:
                        external = tpayload == _TENSOR_LOCATION_EXTERNAL
                    elif tfield == _F_TENSOR_EXTERNAL_DATA and twire == 2:
                        key = value = None
                        for efield, ewire, epayload in _fields(tpayload):
                            if ewire != 2:
                                continue
                            if efield == _F_ENTRY_KEY:
                                key = epayload
                            elif efield == _F_ENTRY_VALUE:
                                value = epayload
                        if key == b"location" and value is not None:
                            location = value.decode("utf-8")
                if external and location and location not in seen:
                    seen.add(location)
                    out.append(os.path.normpath(os.path.join(base, location)))
    except (ValueError, UnicodeDecodeError):
        return []
    return out


class OnnxSession:
    """A loaded fp32 ONNX decision model, run on CPU."""

    def __init__(self, model_path: str, threads: Optional[int] = None):
        # The fp32 export is two files: `model.onnx` (~2.8 MB of graph) and `model.onnx.data`
        # (~1.29 GB of initializers, since the weights exceed protobuf's message ceiling).
        # Deploying only the first is the one mistake this artifact invites, and onnxruntime's
        # own report of it is an external-data error raised from inside tensor loading that
        # never mentions the export. The exporter enforces the pairing at write time; this is
        # the matching check at load time, so the failure names the file and where it comes
        # from. It only ever fires on a location the graph itself declares.
        missing = [p for p in declared_external_data(model_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "%r declares external weight file(s) %s, which are not there. An fp32 export is "
                "a two-file artifact -- the graph in %r and its initializers in the sibling "
                ".data file, which onnx resolves relative to the graph's own directory -- and "
                "both must be copied together. Re-run `python -m laya_onnx.export.export_fp32` "
                "or copy the missing file(s) next to the graph."
                % (model_path, ", ".join(repr(os.path.basename(p)) for p in missing),
                   os.path.basename(model_path)))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None:
            opts.intra_op_num_threads = int(threads)
        self._sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def run(self, batch: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run one collated batch. Returns `(logits, act_logits)` as float32.

        `np.asarray(..., dtype=np.float32)` is a no-op copy-free cast for the graph's own fp32
        outputs; it is here so the return type stays float32 if a later quantized export ever
        hands back something else, rather than letting the dtype leak into postprocessing.
        """
        logits, act = self._sess.run(_OUTPUTS, {k: batch[k] for k in _INPUTS})
        return np.asarray(logits, dtype=np.float32), np.asarray(act, dtype=np.float32)
