"""OnnxAgent: the torch-free equivalent of laya.Agent, for the multilingual (mmBERT) checkpoint.

This module is the seam where Tasks 1-6 become one callable pipeline: TokenizerAdapter ->
build_sequence -> collate -> OnnxSession -> build_answers, wrapped the same way laya.agent.Agent
wraps the torch path (laya/agent.py:266-370), so `OnnxAgent.system_one` returns the same shaped
answer dict for the same inputs.

Two things bind this module and are each covered by tests/test_onnx_runtime.py rather than
merely asserted in a comment:

1. `max_len`/`head_max_len` MUST come from the checkpoint's own `rl_agent_config.json`, never
   from `build_sequence`'s defaults (512/192, upstream's English-checkpoint numbers kept for
   byte-parity). The multilingual checkpoint this port targets runs at 1024/256 -- a ~768-token
   state budget instead of ~317 -- and that budget is the whole reason this port exists on top
   of mmBERT rather than the English checkpoint. Reading `self.cfg.get("max_len", 1024)` with a
   1024 fallback (not 512) means even a checkpoint whose config is missing the key fails toward
   the multilingual number, not silently back toward upstream's English one.

2. A question with fewer than 2 options must be rejected before it reaches the graph. The graph
   was traced with 3 markers (laya_onnx/export/export_fp32.py's `markers = 3` sample) specifically
   because laya/common.py:115 branches on `p.size(-1) >= 2` and only the true side calls
   `p.topk(2, -1)` -- a tracer resolves that Python `if` once, against the sample, and bakes
   whichever branch it took into the graph permanently. A single-option question would feed the
   traced (>=2) branch an input it structurally cannot handle and fail deep inside ONNX Runtime's
   TopK op with a shape error that says nothing about *why*. Reject it here instead, in terms
   the caller can act on.
"""
import json
import os
from typing import Any, Dict, Optional, Union

from .collate import collate
from .postprocess import QTYPES, build_answers
from .sequence import build_sequence, render_options
from .session import OnnxSession
from .tokenizer import TokenizerAdapter
from .truncation import truncation_report


def _to_internal(qdef: Dict) -> Dict:
    """Mirror of laya/agent.py:255 (`Agent._to_internal`). Kept as a free function here since
    OnnxAgent has no torch-side counterpart to inherit it from."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


class OnnxAgent:
    """Loads a laya-onnx checkpoint directory (model.onnx + rl_agent_config.json + tokenizer/)
    and answers typed questions against it without torch or transformers in the process.
    """

    def __init__(self, model_dir: str, threads: Optional[int] = None):
        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                "Incompatible checkpoint: %r does not contain 'rl_agent_config.json'. That file "
                "ships with every laya-onnx export and carries the token budget "
                "(max_len/head_max_len) this runtime must use -- load a directory produced by "
                "laya_onnx.export.export_fp32, not an arbitrary ONNX file." % model_dir
            )
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.tok = TokenizerAdapter(model_dir)
        self.session = OnnxSession(os.path.join(model_dir, "model.onnx"), threads=threads)

        # `laya-multilingual` ships no fitted temperatures at all -- these defaults (identity
        # scaling) are the normal path for that checkpoint, not a fallback for a broken one.
        # See postprocess.py's module docstring: build_answers clamps whatever comes out of
        # these lookups on every call, so an absent/empty mapping here is safe by construction.
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})

    def system_one(self, state: Union[str, dict, list],
                    questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single forward pass. See
        laya.agent.Agent.system_one for the docstring this mirrors; the return shape matches
        `{"model", "answers", "usage"}`, with `usage["truncated"]` as this port's one deliberate
        addition -- the torch path never reports how much of a state it had to drop.
        """
        ids = list(questions.keys())
        # Requirement 1 (see module docstring): read from the checkpoint's own config, and fail
        # toward the multilingual budget (1024/256), never upstream's English default (512/192),
        # if a key happens to be missing.
        max_len = self.cfg.get("max_len", 1024)
        head_max_len = self.cfg.get("head_max_len", 256)

        items, internal = [], {}
        for qid in ids:
            q = _to_internal(questions[qid])
            internal[qid] = q
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            # Requirement 2 (see module docstring): the exported graph was traced with >= 2
            # markers and cannot answer a single-option question at all.
            if len(markers) < 2:
                raise ValueError(
                    "question %r has %d option(s), but the exported ONNX graph was traced with "
                    "laya/common.py's topk(2) branch and requires at least 2 markers per "
                    "question. Give it at least two options, or answer it with a 'noul' "
                    "question instead." % (qid, len(markers))
                )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        batch = collate(items, self.tok.pad_token_id)
        logits, act = self.session.run(batch)
        answers = build_answers(ids, internal, items, logits, act,
                                 self.temperature, self.temperature_by_options)

        truncated = {
            qid: truncation_report(self.tok, state, item["ids"], max_len, internal[qid], head_max_len)
            for qid, item in zip(ids, items)
        }

        return {
            "model": "laya-onnx",
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
                "truncated": truncated,
            },
        }

    predict = system_one


def load(model_dir: str, threads: Optional[int] = None) -> OnnxAgent:
    """Load a laya-onnx checkpoint directory. Mirrors laya.agent.load's shape but takes a local
    path only -- there is no Hub download path here; that is out of scope for this module
    (checkpoint acquisition is Task 8's job)."""
    return OnnxAgent(model_dir, threads=threads)
