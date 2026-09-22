"""Minimal tokenizer adapter: the slice of the HF API that build_sequence needs.

Backed by `tokenizers` (the Rust library) rather than `transformers`, because
transformers imports torch. Special-token ids are resolved from the checkpoint's
own files, never hardcoded -- the checkpoint owns its vocabulary.

Deliberately does NOT read tokenizer_config.json. laya/agent.py:_fix_tokenizer_config
exists because mmBERT/Gemma checkpoints store `extra_special_tokens` as a list where
transformers' AutoTokenizer expects a mapping ("'list' object has no attribute 'keys'"),
and that loader has to rewrite the file on disk before AutoTokenizer can load it at all.
This adapter sidesteps the whole problem by never going near tokenizer_config.json: it
loads tokenizer.json directly via `tokenizers.Tokenizer.from_file` (which only needs the
vocab + merges/model definition) and resolves special-token ids from special_tokens_map.json
plus `Tokenizer.token_to_id`, so the Gemma quirk simply never comes into play here.
"""
import json
import os
from typing import Dict, List

from tokenizers import Tokenizer

_DEFAULTS = {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}


class TokenizerAdapter:
    def __init__(self, model_dir: str):
        tokdir = os.path.join(model_dir, "tokenizer")
        tok_path = os.path.join(tokdir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                "No tokenizer.json in %r. A laya checkpoint ships one under tokenizer/; "
                "run the export first." % tokdir
            )
        self._tok = Tokenizer.from_file(tok_path)
        names = dict(_DEFAULTS)
        smap = os.path.join(tokdir, "special_tokens_map.json")
        if os.path.exists(smap):
            with open(smap, encoding="utf-8") as f:
                loaded = json.load(f)
            for key in _DEFAULTS:
                val = loaded.get(key)
                # transformers writes either "[MASK]" or {"content": "[MASK]", ...}
                if isinstance(val, dict):
                    val = val.get("content")
                if isinstance(val, str):
                    names[key] = val

        self.mask_token = names["mask_token"]
        self.mask_token_id = self._id(names["mask_token"])
        self.cls_token_id = self._id(names["cls_token"])
        self.sep_token_id = self._id(names["sep_token"])
        self.pad_token_id = self._id(names["pad_token"])

    def _id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise ValueError("special token %r is not in this tokenizer's vocabulary" % token)
        return int(tid)

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)}
