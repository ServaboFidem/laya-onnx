"""NumPy collate for the ONNX path, mirroring laya/common.py:247 (`collate_items`) without torch.

`collate_items` batches a list of per-question item dicts (each already tokenized by
`build_sequence`) into padded tensors the model consumes. This is the same padding logic
rewritten against `numpy.ndarray` instead of `torch.Tensor`, with the training-only keys
(`target`, `label`, `meta`) dropped: this runtime only ever runs inference, one call's worth
of items at a time, so there is no group-of-groups batch structure and nothing to attach a
training target to.

`marker_pos`/`marker_mask` follow the model's own convention (laya/common.py:114): positions
past a question's own marker count are left at 0 and masked out, never left as garbage, so a
padded marker never accidentally points at a real token.
"""
from typing import Any, Dict, List

import numpy as np


def collate(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, np.ndarray]:
    if not items:
        raise ValueError("collate() received no items")
    n = len(items)
    seq_len = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)

    ids = np.full((n, seq_len), pad_id, dtype=np.int64)
    att = np.zeros((n, seq_len), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)

    for i, it in enumerate(items):
        seq = it["ids"]
        ids[i, : len(seq)] = seq
        att[i, : len(seq)] = 1
        k = len(it["markers"])
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }
