"""Minimal tokenizer adapter: the slice of the HF API that build_sequence needs.

Backed by `tokenizers` (the Rust library) rather than `transformers`, because
transformers imports torch. Special-token ids are resolved from the checkpoint's
own files, never hardcoded -- the checkpoint owns its vocabulary.

Special-token NAMES are read from both of the files transformers reads them from -
`special_tokens_map.json` and `tokenizer_config.json` - with the same precedence
transformers uses: `tokenizer_config.json` wins, because that is where
`PreTrainedTokenizerBase.from_pretrained` takes its init kwargs from and
`special_tokens_map.json` only fills in what those leave unset.

An earlier version of this adapter read `special_tokens_map.json` only, on the theory that
avoiding `tokenizer_config.json` also avoided the Gemma quirk that
laya/agent.py:_fix_tokenizer_config exists for (mmBERT/Gemma checkpoints store
`extra_special_tokens` as a list where AutoTokenizer wants a mapping, raising "'list' object
has no attribute 'keys'"). That reasoning was wrong twice over, and the real multilingual
checkpoint proved it:

  - `convaiinnovations/laya` @ multilingual ships NO `special_tokens_map.json` at all. Its
    four special tokens live only in `tokenizer_config.json` - cls `<bos>`, sep `<eos>`,
    mask `<mask>`, pad `<pad>`. With that file unread the adapter fell all the way back to
    the BERT-shaped `_DEFAULTS` below and died on `special token '[MASK]' is not in this
    tokenizer's vocabulary`. Falling back silently would have been far worse: `[CLS]`/`[SEP]`
    happening to exist in some other vocabulary would have built sequences with the wrong
    framing tokens and produced confidently wrong answers.
  - the Gemma quirk is not a reason to avoid the file. It breaks AutoTokenizer's
    *constructor*; reading four string fields out of the same JSON cannot trip over it,
    because nothing here ever looks at `extra_special_tokens`.

The vocabulary itself still comes from `tokenizer.json` via `tokenizers.Tokenizer.from_file`,
and ids are still resolved through `token_to_id` - the checkpoint owns its vocabulary and
nothing is hardcoded.
"""
import json
import os
from typing import Dict, List

from tokenizers import Tokenizer

_DEFAULTS = {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}


def _read_token_names(path: str) -> Dict[str, str]:
    """Pull whichever of the four special-token names `path` declares.

    An absent file, an absent key and a value of an unexpected type all mean "declares
    nothing": the caller layers several of these over `_DEFAULTS`, so a partial answer from
    one source is the normal case rather than an error, and a token no source resolves is
    caught once in `_id` after every source has had its say.

    Malformed JSON is deliberately NOT swallowed. A tokenizer directory whose config does not
    parse is a broken checkpoint, and treating it as "declares nothing" would quietly hand the
    adapter the BERT-shaped defaults - the exact silent-wrong-framing-tokens failure this
    module's docstring exists to warn about.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    out = {}
    for key in _DEFAULTS:
        val = loaded.get(key)
        # transformers writes either "[MASK]" or {"content": "[MASK]", ...} (an AddedToken).
        if isinstance(val, dict):
            val = val.get("content")
        if isinstance(val, str):
            out[key] = val
    return out


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
        # Lowest precedence first: _DEFAULTS, then special_tokens_map.json, then
        # tokenizer_config.json last so it wins - see the module docstring for why that order
        # is transformers' order and not an arbitrary choice.
        for fname in ("special_tokens_map.json", "tokenizer_config.json"):
            names.update(_read_token_names(os.path.join(tokdir, fname)))

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
