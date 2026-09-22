"""Report how much of the state build_sequence had to drop.

laya/common.py:82 (and its vendored copy at laya_onnx/sequence.py:82) clips the state to
whatever room is left after the option head, and says nothing about it. That silence is the
failure mode this module exists to remove: a `noul` answer built from half a stack trace looks
exactly like an answer built from the whole thing, and nothing in the model's output tells you
which one you got.

It also turns the reason this port uses mmBERT (`multilingual`, ~1024-token context) instead of
the English checkpoint (~512) into something checkable rather than assumed: a state budget claim
that nobody measures against real inputs is just a comment.

Why this measures the head by rebuilding it, not by arithmetic on `len(seq)`:
`build_sequence` (laya_onnx/sequence.py:47-84) lays a sequence out as
    [CLS] head_ids [SEP] opt0_ids opt1_ids ... [SEP] state_ids [SEP]
where `head_ids` (instructions truncated to `opt_budget`) and the option ids are both functions
of the question and `head_max_len` alone -- never of the state. So for a *fixed* question and
`head_max_len`, the number of ids in `seq` that belong to the head is constant regardless of what
state produced `seq`. The only reliable way to read that constant back out is to ask
`build_sequence` for it directly, with the same question and budgets but an empty state: whatever
length comes back, minus the trailing [SEP] that an empty state still gets appended, is exactly
where the state tokens start in the real `seq`. Subtracting `len(seq)` from that boundary (again
correcting for the real sequence's own trailing [SEP]) gives the count of state tokens that
survived truncation, with no dependence on `len(seq)` for the very quantity it's trying to
isolate -- which is the bug in the brief's original one-line formula.
"""
from typing import Any, Dict, List, Union

from .sequence import build_sequence, serialize_state


def state_token_count(tok, state: Union[str, dict, list]) -> int:
    """Tokens the state would need if nothing were clipped.

    Serializes and mask-token-scrubs the state exactly as build_sequence does before
    tokenizing it (laya_onnx/sequence.py:81), so this count is directly comparable to
    the ids build_sequence actually produces for the same state -- not an approximation
    of them.
    """
    text = serialize_state(state).replace(tok.mask_token, " ")
    return len(tok(text, add_special_tokens=False)["input_ids"])


def truncation_report(
    tok,
    state: Union[str, dict, list],
    seq: List[int],
    max_len: int,
    q: Dict,
    head_max_len: int,
) -> Dict[str, Any]:
    """Account for the state tokens that reached the model versus the ones that did not.

    `seq` must be the id list `build_sequence(tok, state, q, max_len, head_max_len)` returned
    for this exact state -- `q` and `head_max_len` are required (beyond the brief's original
    signature) because the head-length measurement below only isolates the state's contribution
    to `seq` when it is rebuilt against the *same* question and option budget that produced it.
    """
    needed = state_token_count(tok, state)

    # Rebuild the same question's sequence with an empty state. The head is unaffected by the
    # state (see module docstring), so this reveals exactly where the head ends and the state
    # begins in `seq`. An empty state still contributes an (empty) tokenization plus its own
    # trailing [SEP] (laya_onnx/sequence.py:81-83), so `len(empty_ids) - 1` is the boundary.
    empty_ids, _ = build_sequence(tok, "", q, max_len, head_max_len)
    head_len = len(empty_ids) - 1

    # `seq` is head_len head/option ids, followed by however many state ids fit, followed by one
    # trailing [SEP]. Clamp against [0, needed] as a defensive floor/ceiling -- e.g. if a caller
    # passes a `seq` that was built with different arguments than `state`/`q`/`head_max_len`,
    # this keeps the reported numbers sane rather than negative or larger than what was needed.
    used = max(0, min(needed, len(seq) - head_len - 1))
    dropped = max(0, needed - used)

    return {
        "state_tokens": needed,
        "state_tokens_used": used,
        "dropped": dropped,
        "truncated": dropped > 0,
    }
