"""Phase 5: the HTTP server end to end, over real sockets.

Each test starts the app under uvicorn on a free port (httpx's in-process ASGI transport
buffers whole responses, which would hide exactly what is being tested: incremental
streaming and disconnects). Correctness is always checked against the Phase 2 greedy
decode of the same prompt run alone.
"""

import asyncio
import json
import random
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from tokquay.engine import Engine
from tokquay.kv_cache import BlockAllocator
from tokquay.scheduler import SchedulerConfig
from tokquay.server import create_app

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LiveServer:
    """The app under uvicorn on a free port, in a background thread."""

    def __init__(self, model, tokenizer, num_blocks=256, block_size=16, eos_token_id=None, **sched):
        allocator = BlockAllocator(model.cfg, num_blocks, block_size, device=DEVICE)
        extra = {} if eos_token_id is None else {"eos_token_id": eos_token_id}
        self.engine = Engine(model, allocator, SchedulerConfig(**sched) if sched else None, **extra)
        config = uvicorn.Config(create_app(self.engine, tokenizer), host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=30)


@pytest.fixture
def make_server(shared_model, shared_tok):
    started = []

    def make(**kwargs):
        started.append(LiveServer(shared_model, shared_tok, **kwargs).start())
        return started[-1]

    yield make
    for server in reversed(started):
        server.stop()


@pytest.fixture
def server(make_server):
    return make_server()


# ------------------------------------------------------------------------------ helpers
def client_for(server):
    return httpx.AsyncClient(base_url=server.url, timeout=60)


def make_prompt(tok, text_ids, i, n_tokens):
    """(prompt text, the token ids the server will see for it)."""
    text = tok.decode(text_ids[i * 13 : i * 13 + n_tokens])
    return text, tok.encode(text)


def decoded(tok, ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


async def sse(client, body):
    """POST with stream=true; returns [(event name or None, parsed data), ...]."""
    out, name = [], None
    async with client.stream("POST", "/generate", json={**body, "stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                out.append((name, json.loads(line[6:])))
                name = None
    return out


async def wait_until(predicate, timeout=15, what="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


async def wait_idle(client):
    """Until the engine has nothing running or queued and every KV block is back in the pool."""
    last = {}

    async def poll():
        nonlocal last
        last = (await client.get("/health")).json()
        return last["running"] == 0 and last["waiting"] == 0 and last["free_blocks"] == last["num_blocks"]

    deadline = time.monotonic() + 15
    while not await poll():
        if time.monotonic() > deadline:
            pytest.fail(f"engine did not go idle: {last}")
        await asyncio.sleep(0.05)
    return last


# ------------------------------------------------------------------------------ correctness
async def test_non_streaming_response_matches_reference(server, shared_tok, text_ids, greedy_reference):
    prompt, ids = make_prompt(shared_tok, text_ids, 0, 12)
    async with client_for(server) as c:
        r = await c.post("/generate", json={"prompt": prompt, "max_tokens": 25, "ignore_eos": True})
    assert r.status_code == 200
    body, ref = r.json(), greedy_reference(ids, 25)
    assert body["token_ids"] == ref
    assert body["text"] == decoded(shared_tok, ref)
    assert body["finish_reason"] == "length"
    assert body["prompt_tokens"] == len(ids) and body["completion_tokens"] == 25


async def test_streaming_response_matches_reference(server, shared_tok, text_ids, greedy_reference):
    prompt, ids = make_prompt(shared_tok, text_ids, 1, 15)
    async with client_for(server) as c:
        events = await sse(c, {"prompt": prompt, "max_tokens": 25, "ignore_eos": True})
    ref = greedy_reference(ids, 25)
    assert all(name is None for name, _ in events)
    data = [d for _, d in events]
    assert [d["token_id"] for d in data] == ref  # one event per token, in order
    assert "".join(d["text"] for d in data) == decoded(shared_tok, ref)
    assert [d["finish_reason"] for d in data] == [None] * 24 + ["length"]  # only the last one is final


async def test_stream_delivers_tokens_while_generation_is_still_running(server, shared_tok, text_ids):
    prompt, _ = make_prompt(shared_tok, text_ids, 2, 10)
    body = {"prompt": prompt, "max_tokens": 300, "ignore_eos": True, "stream": True}
    async with client_for(server) as c, c.stream("POST", "/generate", json=body) as r:
        first = None
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                first = json.loads(line[6:])
                break
        assert first is not None and first["finish_reason"] is None
        # The request is still generating: the token really was sent as soon as it existed.
        assert (await c.get("/health")).json()["running"] == 1


async def test_twenty_concurrent_clients_get_complete_correct_outputs(server, shared_tok, text_ids, greedy_reference):
    rng = random.Random(7)
    jobs = []
    for i in range(20):
        prompt, ids = make_prompt(shared_tok, text_ids, i, rng.randint(5, 60))
        jobs.append((prompt, ids, rng.randint(8, 24), i % 2 == 0))  # half stream, half do not

    async def one(c, prompt, n, streaming):
        body = {"prompt": prompt, "max_tokens": n, "ignore_eos": True}
        if streaming:
            data = [d for _, d in await sse(c, body)]
            return [d["token_id"] for d in data], "".join(d["text"] for d in data), data[-1]["finish_reason"]
        r = await c.post("/generate", json=body)
        assert r.status_code == 200
        j = r.json()
        return j["token_ids"], j["text"], j["finish_reason"]

    async with client_for(server) as c:
        results = await asyncio.gather(*(one(c, p, n, s) for p, _, n, s in jobs))
        for (prompt, ids, n, _), (token_ids, text, finish) in zip(jobs, results):
            ref = greedy_reference(ids, n)
            assert token_ids == ref
            assert text == decoded(shared_tok, ref)
            assert finish == "length"
        await wait_idle(c)
    assert server.engine.stats.max_batch > 1, "the requests were meant to be batched together"
    assert server.engine.stats.tokens_generated == sum(n for _, _, n, _ in jobs)


async def test_concurrent_clients_on_a_tiny_pool_survive_preemption(make_server, shared_tok, text_ids, greedy_reference):
    server = make_server(num_blocks=12)  # 192 tokens of KV for 20 requests wanting ~40-60 each
    rng = random.Random(11)
    jobs = []
    for i in range(20):
        prompt, ids = make_prompt(shared_tok, text_ids, i, rng.randint(15, 30))
        jobs.append((prompt, ids, rng.randint(20, 32)))

    async with client_for(server) as c:
        responses = await asyncio.gather(
            *(c.post("/generate", json={"prompt": p, "max_tokens": n, "ignore_eos": True}) for p, _, n in jobs)
        )
        for (_, ids, n), r in zip(jobs, responses):
            assert r.status_code == 200
            assert r.json()["token_ids"] == greedy_reference(ids, n)
        health = await wait_idle(c)
    assert server.engine.stats.num_preemptions > 0, "the pool was meant to be too small"
    assert health["preemptions"] == server.engine.stats.num_preemptions


async def test_request_arriving_mid_stream_finishes_first_without_waiting_for_the_long_one(
    server, shared_tok, text_ids, greedy_reference
):
    long_prompt, _ = make_prompt(shared_tok, text_ids, 3, 10)
    short_prompt, short_ids = make_prompt(shared_tok, text_ids, 4, 10)
    long_body = {"prompt": long_prompt, "max_tokens": 300, "ignore_eos": True, "stream": True}
    async with client_for(server) as c, c.stream("POST", "/generate", json=long_body) as long_stream:
        lines = long_stream.aiter_lines()
        async for line in lines:  # the long request is now generating
            if line.startswith("data: "):
                break
        r = await c.post("/generate", json={"prompt": short_prompt, "max_tokens": 6, "ignore_eos": True})
        assert r.json()["token_ids"] == greedy_reference(short_ids, 6)
        assert (await c.get("/health")).json()["running"] == 1  # the long one is still going


async def test_seeded_sampling_is_reproducible_over_http(server, shared_tok, text_ids, greedy_reference):
    prompt, ids = make_prompt(shared_tok, text_ids, 5, 10)
    sampled = {"prompt": prompt, "max_tokens": 20, "temperature": 0.9, "top_k": 40, "ignore_eos": True}

    async with client_for(server) as c:
        a, b, other, greedy = await asyncio.gather(
            *(
                c.post("/generate", json=body)
                for body in (
                    {**sampled, "seed": 1234},
                    {**sampled, "seed": 1234},
                    {**sampled, "seed": 99},
                    {"prompt": prompt, "max_tokens": 20, "ignore_eos": True},
                )
            )
        )
    a, b, other, greedy = (r.json()["token_ids"] for r in (a, b, other, greedy))
    assert a == b  # same seed, same text, even though the requests shared a batch
    assert a != other and a != greedy
    assert greedy == greedy_reference(ids, 20)


async def test_eos_ends_generation_with_finish_reason_stop(make_server, shared_tok, text_ids, greedy_reference):
    prompt, ids = make_prompt(shared_tok, text_ids, 6, 10)
    ref = greedy_reference(ids, 12)
    i = next(i for i in range(1, 12) if ref[i] not in ref[:i])
    server = make_server(eos_token_id=ref[i])  # pretend this token is the end-of-text marker
    async with client_for(server) as c:
        r = await c.post("/generate", json={"prompt": prompt, "max_tokens": 12})
        events = await sse(c, {"prompt": prompt, "max_tokens": 12})
    assert r.json()["token_ids"] == ref[: i + 1] and r.json()["finish_reason"] == "stop"
    assert [d["token_id"] for _, d in events] == ref[: i + 1] and events[-1][1]["finish_reason"] == "stop"


# ------------------------------------------------------------------------------ disconnects
async def test_streaming_client_disconnect_aborts_the_request_and_frees_its_blocks(server, shared_tok, text_ids):
    prompt, _ = make_prompt(shared_tok, text_ids, 7, 10)
    body = {"prompt": prompt, "max_tokens": 400, "ignore_eos": True, "stream": True}
    async with client_for(server) as c:
        async with c.stream("POST", "/generate", json=body) as r:
            seen = 0
            async for line in r.aiter_lines():
                seen += line.startswith("data: ")
                if seen == 3:
                    break
            seq = server.engine.scheduler.running[0]
            assert seq.output_token_ids
        # leaving the block closed the connection mid-stream
        await wait_idle(c)
    assert seq.finish_reason == "abort"
    assert len(seq.output_token_ids) < 400  # it did not keep generating for nobody


async def test_non_streaming_client_disconnect_aborts_the_request_and_frees_its_blocks(server, shared_tok, text_ids):
    prompt, _ = make_prompt(shared_tok, text_ids, 8, 10)
    async with client_for(server) as c, client_for(server) as watcher:
        call = asyncio.create_task(c.post("/generate", json={"prompt": prompt, "max_tokens": 400, "ignore_eos": True}))
        await wait_until(lambda: server.engine.scheduler.running, what="the request to start running")
        seq = server.engine.scheduler.running[0]
        call.cancel()  # the client gives up and drops the connection
        with pytest.raises(asyncio.CancelledError):
            await call
        await wait_idle(watcher)
    assert seq.finish_reason == "abort"
    assert len(seq.output_token_ids) < 400


# ------------------------------------------------------------------------------ errors
async def test_unservable_and_malformed_requests_are_refused_and_the_server_keeps_working(
    make_server, shared_tok, text_ids, greedy_reference
):
    server = make_server(num_blocks=4)  # 64 tokens of KV
    prompt, ids = make_prompt(shared_tok, text_ids, 9, 10)
    async with client_for(server) as c:
        too_long_for_pool = await c.post("/generate", json={"prompt": "hello " * 100, "max_tokens": 4})
        assert too_long_for_pool.status_code == 400 and "KV blocks" in too_long_for_pool.json()["detail"]
        too_long_for_model = await c.post("/generate", json={"prompt": "hello " * 1100, "max_tokens": 4})
        assert too_long_for_model.status_code == 400 and "positions" in too_long_for_model.json()["detail"]

        for bad in (
            {"prompt": ""},
            {"max_tokens": 5},  # no prompt
            {"prompt": "hi", "max_tokens": 0},
            {"prompt": "hi", "temperature": -0.5},
            {"prompt": "hi", "top_k": -1},
            {"prompt": "hi", "max_tokens": "many"},
        ):
            assert (await c.post("/generate", json=bad)).status_code == 422, bad

        ok = await c.post("/generate", json={"prompt": prompt, "max_tokens": 8, "ignore_eos": True})
        assert ok.status_code == 200 and ok.json()["token_ids"] == greedy_reference(ids, 8)
        await wait_idle(c)


async def test_health_reports_pool_and_queue_state(server):
    async with client_for(server) as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "running": 0,
        "waiting": 0,
        "free_blocks": 256,
        "num_blocks": 256,
        "preemptions": 0,
    }


async def test_engine_crash_fails_requests_instead_of_leaving_them_hanging(make_server, shared_tok, text_ids):
    server = make_server()
    real_step, calls = server.engine.step, 0

    def flaky_step():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("boom")
        return real_step()

    server.engine.step = flaky_step
    prompt, _ = make_prompt(shared_tok, text_ids, 10, 10)
    body = {"prompt": prompt, "max_tokens": 100, "ignore_eos": True}
    async with client_for(server) as c:
        events, plain = await asyncio.wait_for(asyncio.gather(sse(c, body), c.post("/generate", json=body)), 30)
        name, payload = events[-1]
        assert name == "error" and "boom" in payload["error"]  # the stream ends with an error event
        assert all(n is None for n, _ in events[:-1])
        assert plain.status_code == 503  # the plain request fails too

        assert (await c.post("/generate", json=body)).status_code == 503  # and new ones are refused
        health = await c.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "failed"
