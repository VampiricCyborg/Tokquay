"""Phase 5: streaming text must equal the batch decode and never show a half character."""

from tokquay.detokenizer import REPLACEMENT, IncrementalDetokenizer

EOS = 50256


def stream(tok, ids, flush=True):
    d = IncrementalDetokenizer(tok)
    pieces = [d.push(i) for i in ids]
    if flush:
        pieces.append(d.flush())
    return pieces


def full_decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def test_ascii_deltas_concatenate_to_the_full_decode(shared_tok, text_ids):
    ids = text_ids[:120]
    pieces = stream(shared_tok, ids)
    assert "".join(pieces) == full_decode(shared_tok, ids)
    assert all(pieces[:-1])  # plain words: every token yields text straight away


def test_multibyte_characters_are_held_back_until_complete(shared_tok):
    text = "café \U0001f642 世界 \U0001f680 naïve — done"
    ids = shared_tok.encode(text)
    pieces = stream(shared_tok, ids)
    assert "".join(pieces) == text
    assert not any(REPLACEMENT in p for p in pieces)  # never a half character
    assert "" in pieces[:-1], "expected some token to be a partial character that gets held back"


def test_flush_emits_an_incomplete_character_at_end_of_stream(shared_tok):
    ids = shared_tok.encode("ok \U0001f680")  # the rocket is spread over several byte tokens
    assert len(ids) >= 3
    cut = ids[:-1]  # drop the last byte: the character is now incomplete
    assert full_decode(shared_tok, cut).endswith(REPLACEMENT)
    unflushed = stream(shared_tok, cut, flush=False)
    assert not any(REPLACEMENT in p for p in unflushed)
    pieces = stream(shared_tok, cut)
    assert "".join(pieces) == full_decode(shared_tok, cut)  # the tail arrives on flush


def test_end_of_text_token_is_not_emitted(shared_tok):
    ids = shared_tok.encode("hello world") + [EOS]
    pieces = stream(shared_tok, ids)
    assert "".join(pieces) == "hello world"
    assert pieces[-2] == ""  # the EOS token itself produced no text


def test_flush_on_an_empty_or_complete_stream_is_empty(shared_tok):
    assert IncrementalDetokenizer(shared_tok).flush() == ""
    d = IncrementalDetokenizer(shared_tok)
    for i in shared_tok.encode("finished"):
        d.push(i)
    assert d.flush() == ""
