"""Phase 5: FastAPI server with SSE streaming.

``POST /generate {prompt, max_tokens, temperature, top_k, stream}``

* ``stream: false`` (default): one JSON body when the request finishes::

      {"text": "...", "token_ids": [...], "finish_reason": "length",
       "prompt_tokens": 12, "completion_tokens": 25}

* ``stream: true``: ``text/event-stream``, one event per generated token::

      data: {"token_id": 464, "text": " the", "finish_reason": null}

  ``text`` is the newly completed text (empty while a multi-byte character is still
  incomplete). The last event has a non-null ``finish_reason``. If the engine dies
  mid-stream the last event is ``event: error``.

Requests that can never be served (too long for the model or the KV pool) get a 400
up front; schema violations get a 422; a dead engine gets a 503.

``GET /health`` reports the queue lengths and the free KV blocks.
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from tokquay.async_engine import AsyncEngine, EngineError, RequestStream
from tokquay.detokenizer import IncrementalDetokenizer
from tokquay.engine import Engine
from tokquay.kv_cache import BlockAllocator
from tokquay.model import GPT2
from tokquay.scheduler import SchedulerConfig
from tokquay.sequence import SamplingParams


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(0.0, ge=0)  # 0 = greedy
    top_k: int = Field(0, ge=0)  # 0 = no top-k filtering
    stream: bool = False
    seed: int | None = None  # makes sampled output reproducible
    ignore_eos: bool = False  # keep generating past <|endoftext|> (fixed-length runs)


def _sse(payload: dict, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(payload)}\n\n"


async def _stream_events(stream: RequestStream, tokenizer):
    detok = IncrementalDetokenizer(tokenizer)
    try:
        async for ev in stream:
            text = detok.push(ev.token_id)
            if ev.finish_reason is not None:
                text += detok.flush()
            yield _sse({"token_id": ev.token_id, "text": text, "finish_reason": ev.finish_reason})
    except EngineError as exc:
        yield _sse({"error": str(exc)}, event="error")
    finally:
        stream.abort()  # a client that hung up must not keep the GPU busy; no-op if finished


async def _collect(stream: RequestStream, request: Request, tokenizer) -> Response:
    detok = IncrementalDetokenizer(tokenizer)
    token_ids: list[int] = []
    pieces: list[str] = []
    finish_reason = None
    try:
        async for ev in stream:
            token_ids.append(ev.token_id)
            pieces.append(detok.push(ev.token_id))
            finish_reason = ev.finish_reason
            # A plain (non-streaming) handler is not cancelled when the client leaves,
            # so look for the disconnect ourselves once per token.
            if finish_reason is None and await request.is_disconnected():
                return Response(status_code=499)
        pieces.append(detok.flush())
    except EngineError as exc:
        raise HTTPException(503, str(exc))
    finally:
        stream.abort()
    return JSONResponse(
        {
            "text": "".join(pieces),
            "token_ids": token_ids,
            "finish_reason": finish_reason,
            "prompt_tokens": len(stream.prompt_token_ids),
            "completion_tokens": len(token_ids),
        }
    )


def create_app(engine: Engine, tokenizer) -> FastAPI:
    async_engine = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await async_engine.start()
        try:
            yield
        finally:
            await async_engine.stop()

    app = FastAPI(title="Tokquay", lifespan=lifespan)
    app.state.async_engine = async_engine

    @app.post("/generate")
    async def generate(req: GenerateRequest, request: Request):
        try:
            params = SamplingParams(
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                ignore_eos=req.ignore_eos,
                seed=req.seed,
            )
            stream = async_engine.submit(tokenizer.encode(req.prompt), params)
        except EngineError as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:  # includes RequestRejected
            raise HTTPException(400, str(exc))

        if req.stream:
            return StreamingResponse(
                _stream_events(stream, tokenizer),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _collect(stream, request, tokenizer)

    @app.get("/health")
    async def health():
        failed = async_engine.failure is not None
        body = {"status": "failed" if failed else "ok", **async_engine.stats()}
        return JSONResponse(body, status_code=503 if failed else 200)

    return app


def build_engine(
    num_blocks: int = 1024,
    block_size: int = 16,
    device: str | None = None,
    config: SchedulerConfig | None = None,
):
    """Load GPT-2 and allocate the KV pool. Returns ``(engine, tokenizer)``."""
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2.from_pretrained("gpt2").to(device).eval()
    allocator = BlockAllocator(model.cfg, num_blocks, block_size, device=device)
    return Engine(model, allocator, config), AutoTokenizer.from_pretrained("gpt2")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve GPT-2 with continuous batching and a paged KV cache.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--num-blocks", type=int, default=1024, help="KV pool size in blocks")
    ap.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-fraction", type=float, default=None,
                    help="cap PyTorch's CUDA allocator at this fraction of VRAM. On a small Windows GPU (4 GiB) pass 0.8: "
                         "uncapped, the allocator can grow to the whole card and decode steps slow down 10x")  # fmt: skip
    args = ap.parse_args(argv)
    if args.gpu_memory_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    config = SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
    engine, tokenizer = build_engine(args.num_blocks, args.block_size, args.device, config)
    a = engine.allocator
    print(f"KV pool: {a.num_blocks} blocks x {a.block_size} tokens = {a.num_blocks * a.bytes_per_block / 2**20:.0f} MiB")
    uvicorn.run(create_app(engine, tokenizer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
