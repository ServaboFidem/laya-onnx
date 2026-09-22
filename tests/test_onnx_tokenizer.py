"""TokenizerAdapter exposes the slice of the HF tokenizer API build_sequence needs,
backed by `tokenizers` alone so the runtime never imports transformers.

Several cases beyond the brief close a spec gap around special-token resolution, which is
where this adapter met the real multilingual checkpoint and lost:

  - the adapter must read special-token names from BOTH special_tokens_map.json and
    tokenizer_config.json, with tokenizer_config.json winning, because that is transformers'
    own precedence and because `convaiinnovations/laya` @ multilingual ships no
    special_tokens_map.json at all -- its `<bos>`/`<eos>`/`<mask>`/`<pad>` live only in
    tokenizer_config.json. Reading only the map file left the adapter on the BERT-shaped
    defaults and it died on "special token '[MASK]' is not in this tokenizer's vocabulary".
  - reading tokenizer_config.json must NOT reintroduce the Gemma quirk that breaks
    transformers' AutoTokenizer for mmBERT/Gemma checkpoints (`extra_special_tokens` as a
    list -> "'list' object has no attribute 'keys'", see laya/agent.py:39-45). Nothing here
    looks at that key, so it cannot; the test below holds that line.
  - both shapes of a declared token -- bare string and {"content": ..., ...} dict -- resolve.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer                    # noqa: E402
from tokenizers.models import WordLevel             # noqa: E402
from tokenizers.pre_tokenizers import Whitespace    # noqa: E402

from laya_onnx.sequence import build_sequence       # noqa: E402
from laya_onnx.tokenizer import TokenizerAdapter    # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def make_vocab_tokenizer(tokdir):
    """Write tokenizer.json for a minimal word-level vocabulary shared by all fixtures."""
    os.makedirs(tokdir)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i, w in enumerate(["billing", "refund", "charge", "question", "choice", "state", "true", "false"]):
        vocab[w] = 5 + i
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(os.path.join(tokdir, "tokenizer.json"))


def make_checkpoint(tmp):
    """Write a minimal checkpoint layout: <tmp>/tokenizer/{tokenizer.json,special_tokens_map.json}."""
    tokdir = os.path.join(tmp, "tokenizer")
    make_vocab_tokenizer(tokdir)
    with open(os.path.join(tokdir, "special_tokens_map.json"), "w") as f:
        json.dump({"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]",
                   "sep_token": "[SEP]", "mask_token": "[MASK]"}, f)
    return tmp


def make_dict_form_checkpoint(tmp):
    """special_tokens_map.json using the dict form transformers also writes:
    {"mask_token": {"content": "[MASK]", "lstrip": ..., ...}} instead of a bare string.
    The adapter's `isinstance(val, dict)` branch must unwrap "content" from this shape.
    """
    tokdir = os.path.join(tmp, "tokenizer")
    make_vocab_tokenizer(tokdir)

    def wrap(content):
        return {"content": content, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}

    with open(os.path.join(tokdir, "special_tokens_map.json"), "w") as f:
        json.dump({
            "pad_token": wrap("[PAD]"),
            "unk_token": wrap("[UNK]"),
            "cls_token": wrap("[CLS]"),
            "sep_token": wrap("[SEP]"),
            "mask_token": wrap("[MASK]"),
        }, f)
    return tmp


def make_gemma_quirk_checkpoint(tmp):
    """Reproduces the exact tokenizer_config.json shape that breaks transformers'
    AutoTokenizer for mmBERT/Gemma checkpoints: `extra_special_tokens` as a LIST, not a
    mapping (laya/agent.py:39-45 says transformers raises
    "'list' object has no attribute 'keys'" on this shape, which is why laya's own loader
    has to rewrite the file before AutoTokenizer ever sees it).

    TokenizerAdapter must construct and tokenize correctly against this checkpoint WITHOUT
    that rewrite step -- because it never reads tokenizer_config.json at all. This is the
    thing that proves the sidestep, rather than merely assuming it because the brief's
    fixture never wrote the file in the first place.
    """
    tokdir = os.path.join(tmp, "tokenizer")
    make_vocab_tokenizer(tokdir)
    with open(os.path.join(tokdir, "special_tokens_map.json"), "w") as f:
        json.dump({"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]",
                   "sep_token": "[SEP]", "mask_token": "[MASK]"}, f)
    # The quirky config: a list of extra special tokens, plus no explicit tokenizer_class,
    # mirroring what a real mmBERT/Gemma checkpoint ships and what _fix_tokenizer_config
    # exists to repair for transformers. TokenizerAdapter must not need that repair.
    with open(os.path.join(tokdir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": None,
            "extra_special_tokens": ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"],
        }, f)
    return tmp


# --- Case: brief's baseline (bare-string special_tokens_map.json) ---------------------
tmp = make_checkpoint(tempfile.mkdtemp())
try:
    tok = TokenizerAdapter(tmp)

    check("special/mask_token", tok.mask_token, "[MASK]")
    check("special/mask_id", tok.mask_token_id, 4)
    check("special/cls_id", tok.cls_token_id, 2)
    check("special/sep_id", tok.sep_token_id, 3)
    check("special/pad_id", tok.pad_token_id, 0)

    check("encode/known", tok("billing refund")["input_ids"], [5, 6])
    check("encode/unknown-maps-to-unk", tok("zzzz")["input_ids"], [1])
    check("encode/empty", tok("")["input_ids"], [])
    # add_special_tokens=False must not inject CLS/SEP -- build_sequence adds them itself.
    check("encode/no-special-tokens", tok("billing", add_special_tokens=False)["input_ids"], [5])

    # The adapter must be a drop-in for build_sequence.
    q = {"t": "noul", "ins": "refund question", "crit": None}
    ids, markers = build_sequence(tok, "charge state", q, 64, 32)
    check("build_sequence/starts-with-cls", ids[0], 2)
    check("build_sequence/ends-with-sep", ids[-1], 3)
    check("build_sequence/two-markers", len(markers), 2)
    check("build_sequence/markers-are-mask", [ids[m] for m in markers], [4, 4])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- Case: dict-form special_tokens_map.json ({"content": ..., ...}) ------------------
tmp2 = make_dict_form_checkpoint(tempfile.mkdtemp())
try:
    tok2 = TokenizerAdapter(tmp2)
    check("dict-form/mask_token", tok2.mask_token, "[MASK]")
    check("dict-form/mask_id", tok2.mask_token_id, 4)
    check("dict-form/cls_id", tok2.cls_token_id, 2)
    check("dict-form/sep_id", tok2.sep_token_id, 3)
    check("dict-form/pad_id", tok2.pad_token_id, 0)
    check("dict-form/encode", tok2("billing refund")["input_ids"], [5, 6])
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

# --- Case: Gemma tokenizer_config.json quirk (extra_special_tokens as a list) ---------
tmp3 = make_gemma_quirk_checkpoint(tempfile.mkdtemp())
try:
    # Constructing must not raise. The adapter does read tokenizer_config.json, but only for
    # the four special-token names -- it never touches `extra_special_tokens`, so the list
    # shape that makes transformers' AutoTokenizer blow up with "'list' object has no
    # attribute 'keys'" is simply not on any path it walks.
    tok3 = TokenizerAdapter(tmp3)
    check("gemma-quirk/mask_token", tok3.mask_token, "[MASK]")
    check("gemma-quirk/mask_id", tok3.mask_token_id, 4)
    check("gemma-quirk/cls_id", tok3.cls_token_id, 2)
    check("gemma-quirk/sep_id", tok3.sep_token_id, 3)
    check("gemma-quirk/pad_id", tok3.pad_token_id, 0)
    check("gemma-quirk/encode", tok3("billing refund charge")["input_ids"], [5, 6, 7])

    q3 = {"t": "noul", "ins": "refund question", "crit": None}
    ids3, markers3 = build_sequence(tok3, "charge state", q3, 64, 32)
    check("gemma-quirk/build_sequence-starts-with-cls", ids3[0], 2)
    check("gemma-quirk/build_sequence-ends-with-sep", ids3[-1], 3)
finally:
    shutil.rmtree(tmp3, ignore_errors=True)

# ------------------------------------------------------------------ special-token sourcing
# Regression cases from Task 8: the real multilingual checkpoint ships tokenizer.json and
# tokenizer_config.json and nothing else, so special_tokens_map.json cannot be the only source.

# (a) names declared ONLY in tokenizer_config.json must resolve.
tmp4 = tempfile.mkdtemp()
try:
    tokdir4 = os.path.join(tmp4, "tokenizer")
    make_vocab_tokenizer(tokdir4)
    with open(os.path.join(tokdir4, "tokenizer_config.json"), "w") as f:
        json.dump({"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]",
                   "mask_token": "[MASK]", "tokenizer_class": "PreTrainedTokenizerFast"}, f)
    tok4 = TokenizerAdapter(tmp4)
    check("config-only/mask_id", tok4.mask_token_id, 4)
    check("config-only/cls_id", tok4.cls_token_id, 2)
    check("config-only/sep_id", tok4.sep_token_id, 3)
    check("config-only/pad_id", tok4.pad_token_id, 0)
finally:
    shutil.rmtree(tmp4, ignore_errors=True)

# (b) when both files declare a token, tokenizer_config.json wins -- transformers takes its
# init kwargs from tokenizer_config.json and lets special_tokens_map.json fill only the gaps.
# Getting this backwards would frame every sequence with the wrong token on any checkpoint
# where the two files disagree, silently.
tmp5 = tempfile.mkdtemp()
try:
    tokdir5 = os.path.join(tmp5, "tokenizer")
    make_vocab_tokenizer(tokdir5)
    with open(os.path.join(tokdir5, "special_tokens_map.json"), "w") as f:
        json.dump({"cls_token": "[SEP]", "sep_token": "[SEP]", "pad_token": "[PAD]"}, f)
    with open(os.path.join(tokdir5, "tokenizer_config.json"), "w") as f:
        # disagrees with the map on cls_token, and is the only source for mask_token
        json.dump({"cls_token": "[CLS]", "mask_token": "[MASK]"}, f)
    tok5 = TokenizerAdapter(tmp5)
    check("precedence/config-wins-on-cls", tok5.cls_token_id, 2)
    check("precedence/config-supplies-mask", tok5.mask_token_id, 4)
    check("precedence/map-fills-the-gap-sep", tok5.sep_token_id, 3)
    check("precedence/map-fills-the-gap-pad", tok5.pad_token_id, 0)
finally:
    shutil.rmtree(tmp5, ignore_errors=True)

# (c) a name no source can resolve is still fatal. Silently falling back to some other token
# is the failure mode this whole area exists to prevent: it produces sequences framed with
# the wrong tokens and answers that are confidently wrong, with no error anywhere.
tmp6 = tempfile.mkdtemp()
try:
    tokdir6 = os.path.join(tmp6, "tokenizer")
    make_vocab_tokenizer(tokdir6)
    with open(os.path.join(tokdir6, "tokenizer_config.json"), "w") as f:
        json.dump({"mask_token": "<not-in-this-vocab>"}, f)
    raised = ""
    try:
        TokenizerAdapter(tmp6)
    except ValueError as exc:
        raised = str(exc)
    check("unresolvable/raises-ValueError", raised.startswith("special token"), True)
    check("unresolvable/names-the-token", "<not-in-this-vocab>" in raised, True)
finally:
    shutil.rmtree(tmp6, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all tokenizer-adapter tests passed")
sys.exit(1 if FAIL else 0)
