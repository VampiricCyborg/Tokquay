"""Phase 5: turn a stream of token ids into text deltas for streaming.

GPT-2 uses byte-level BPE, so one character can be spread over several tokens
(an emoji is usually 3-4 byte tokens). Decoding each token on its own would emit
U+FFFD ("replacement character") for the incomplete pieces. Instead we keep a
small window of recent ids, decode it before and after the newest token, and emit
only the text that was added. If the window ends in U+FFFD the character is not
complete yet, so nothing is emitted until more tokens arrive.

Byte-level decoding is context-free (a token is a fixed byte string), so cutting
the stream at character boundaries and decoding the pieces separately gives
exactly the same text as decoding everything at once.
"""

from __future__ import annotations

REPLACEMENT = "�"


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._prefix = 0  # ids before this index are fully emitted and no longer needed
        self._read = 0  # ids before this index have had their text emitted

    def _decode(self, ids: list[int]) -> str:
        # skip_special_tokens hides <|endoftext|>; no cleanup, because it depends on
        # context and would make windowed decoding differ from decoding everything.
        return self._tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def push(self, token_id: int) -> str:
        """Add one token; returns the newly completed text (possibly empty)."""
        self._ids.append(token_id)
        return self._emit(final=False)

    def flush(self) -> str:
        """End of stream: emit whatever is held back, even an incomplete character."""
        return self._emit(final=True)

    def _emit(self, final: bool) -> str:
        window = self._ids[self._prefix :]
        before = self._decode(self._ids[self._prefix : self._read])
        after = self._decode(window)
        if len(after) <= len(before) or (after.endswith(REPLACEMENT) and not final):
            return ""
        self._prefix, self._read = self._read, len(self._ids)
        return after[len(before) :]
