"""Phase 5: drive the synchronous engine from asyncio.

``Engine.step()`` is a blocking forward pass. Run on the event loop it would freeze
every open SSE stream and every new connection for the length of the step, so the
loop runs on one dedicated thread instead (PyTorch releases the GIL inside its ops,
so the event loop stays responsive while a step runs).

That thread is the *only* one that ever touches the engine, scheduler or allocator.
The event loop talks to it through an inbox queue ("add this request", "abort that
one") and gets results back through one ``asyncio.Queue`` per request, filled with
``loop.call_soon_threadsafe``. No locks are needed because nothing is shared.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

from tokquay.engine import Engine, TokenEvent
from tokquay.scheduler import RequestRejected
from tokquay.sequence import SamplingParams, Sequence

log = logging.getLogger(__name__)

_STOP = object()  # inbox sentinel: shut the loop down


class EngineError(RuntimeError):
    """The engine loop is not running (it crashed or was stopped)."""


class RequestStream:
    """One in-flight request, as seen from the asyncio side.

    ``async for event in stream`` yields ``TokenEvent``s and ends after the one
    that carries a ``finish_reason``. If the engine dies first, iteration raises
    ``EngineError``. Call ``abort()`` when the consumer goes away early.
    """

    def __init__(self, owner: AsyncEngine, prompt_token_ids: list[int], params: SamplingParams):
        self._owner = owner
        self.prompt_token_ids = prompt_token_ids
        self.params = params
        self._queue: asyncio.Queue[TokenEvent | Exception] = asyncio.Queue()
        self._done = False
        self.seq: Sequence | None = None  # set by the engine thread once the request is queued

    def __aiter__(self) -> RequestStream:
        return self

    async def __anext__(self) -> TokenEvent:
        if self._done:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, Exception):
            self._done = True
            raise item
        if item.finish_reason is not None:
            self._done = True
        return item

    def abort(self) -> None:
        """Stop generating for this request and free its KV blocks. Idempotent, non-blocking."""
        if not self._done:
            self._done = True
            self._owner._post(("abort", self))



def _deliver(pairs: list[tuple[RequestStream, TokenEvent | Exception]]) -> None:
    """Runs on the event loop: hand items to their request's queue."""
    for stream, item in pairs:
        stream._queue.put_nowait(item)


class AsyncEngine:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._streams: dict[int, RequestStream] = {}  # seq_id -> stream; engine thread only
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False  # the engine thread has exited
        self.failure: BaseException | None = None  # why it exited, if it crashed

    # ---- lifecycle -----------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="tokquay-engine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        """Stop the loop after the step in flight; unfinished requests get ``EngineError``."""
        if self._thread is not None:
            self._post(_STOP)
            await asyncio.to_thread(self._thread.join)

    # ---- requests (event loop side) ------------------------------------------------
    def submit(self, prompt_token_ids: list[int], params: SamplingParams) -> RequestStream:
        """Queue a request. Raises ``ValueError`` (``RequestRejected`` if it can never
        be served, plain ``ValueError`` for an empty prompt) or ``EngineError``."""
        if self._closed or self._thread is None:
            raise EngineError(self._why_closed())
        # The rejection rules only depend on constants (pool size, limits), so they can
        # be checked here without touching engine state owned by the other thread.
        probe = Sequence(-1, list(prompt_token_ids), sampling=params)
        reason = self.engine.scheduler.rejection_reason(probe)
        if reason:
            raise RequestRejected(reason)
        stream = RequestStream(self, list(prompt_token_ids), params)
        self._post(("add", stream))
        if self._closed:  # the engine thread exited while we were enqueueing
            self._send([(stream, EngineError(self._why_closed()))])
        return stream

    def stats(self) -> dict[str, int]:
        """A racy-but-safe snapshot (plain int reads) for /health."""
        sched, alloc = self.engine.scheduler, self.engine.allocator
        return {
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "free_blocks": alloc.num_free_blocks,
            "num_blocks": alloc.num_blocks,
            "preemptions": sched.num_preemptions,
        }

    # ---- the engine thread ---------------------------------------------------------
    def _post(self, command) -> None:
        self._inbox.put(command)

    def _run(self) -> None:
        engine = self.engine
        try:
            while True:
                commands = []
                if not engine.has_unfinished():
                    commands.append(self._inbox.get())  # idle: sleep until something arrives
                while True:
                    try:
                        commands.append(self._inbox.get_nowait())
                    except queue.Empty:
                        break
                for command in commands:
                    if command is _STOP:
                        return
                    self._apply(command)
                if engine.has_unfinished():
                    self._publish(engine.step())
        except BaseException as exc:  # a failed step leaves the engine inconsistent: no recovery
            self.failure = exc
            log.exception("engine loop crashed")
        finally:
            self._closed = True
            self._fail_everything(EngineError(self._why_closed()))

    def _apply(self, command) -> None:
        kind, stream = command
        if kind == "add":
            try:
                seq = self.engine.add_request(stream.prompt_token_ids, stream.params)
            except ValueError as exc:  # rejected: tell this request only, the engine is fine
                self._send([(stream, exc)])
                return
            stream.seq = seq
            self._streams[seq.seq_id] = stream
        elif kind == "abort":
            # Commands are FIFO, so an "add" is always applied before its "abort".
            if stream.seq is not None and self._streams.pop(stream.seq.seq_id, None) is not None:
                self.engine.abort_request(stream.seq)

    def _publish(self, events: list[TokenEvent]) -> None:
        pairs = []
        for ev in events:
            stream = self._streams[ev.seq_id]
            if ev.finish_reason is not None:
                del self._streams[ev.seq_id]
            pairs.append((stream, ev))
        self._send(pairs)

    def _send(self, pairs: list[tuple[RequestStream, TokenEvent | Exception]]) -> None:
        """One wake-up of the event loop per step, not one per token."""
        if not pairs:
            return
        try:
            self._loop.call_soon_threadsafe(_deliver, pairs)
        except RuntimeError:  # the loop is already closed (shutting down): nobody is listening
            pass

    def _fail_everything(self, error: EngineError) -> None:
        """No request may be left waiting forever on an engine that is gone."""
        pairs = [(stream, error) for stream in self._streams.values()]
        self._streams.clear()
        while True:
            try:
                command = self._inbox.get_nowait()
            except queue.Empty:
                break
            if command is not _STOP and command[0] == "add":
                pairs.append((command[1], error))
        self._send(pairs)

    def _why_closed(self) -> str:
        return f"engine stopped: {self.failure!r}" if self.failure is not None else "engine stopped"
