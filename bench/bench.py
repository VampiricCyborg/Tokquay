"""Benchmark Tokquay against the static-batching baseline.

    python -m bench.bench calibrate                      # saturated throughput of each system
    python -m bench.bench sweep --rates 0.4,0.8,1.6      # the headline comparison
    python -m bench.bench blocksize --rates 0.8,1.6      # block size 16 vs 32
    python -m bench.bench pool --rate 1.6                # shrink the KV memory budget
    python -m bench.bench report                         # tables + plots from the saved results

Both systems get the *same KV memory budget* (``--budget-tokens`` token slots, 72 KiB
each for GPT-2 small in fp32): Tokquay turns it into ``budget / block_size`` blocks, the
baseline into ``budget / max_len`` rows of ``max_len`` slots. Same model, same request
trace (same seed, so the very same prompts and lengths at every rate), fp32 on one GPU.

Results are written to ``bench/results/results.json`` after every run, so a long
experiment that is interrupted keeps what it finished.

CPU placement: a decode step at these sizes is bound by how fast the Python thread can
launch kernels, so it runs at the speed of whichever core it lands on. On a hybrid CPU
(performance + efficiency cores) Windows moves it around, and the same work measured 2x
apart (see design-notes.md). ``--cpu-affinity`` (a hex mask of logical CPUs, e.g. ``FF``
for the performance cores of an i5-12450H) and ``--high-priority`` pin the process for
every run, and both are recorded in the results.

GPU memory: the PyTorch caching allocator is capped at ``--memory-fraction`` of VRAM
(default 0.8). Uncapped, on a 4 GiB Windows card it fragments its way to the whole card
during the paged gather at high concurrency, the driver starts spilling to system memory,
and some decode steps take seconds instead of 55 ms (see design-notes.md). The cap is
applied to both systems, and every run records its allocator health so this cannot
silently happen again.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import platform
import time
from pathlib import Path

import torch

from bench.harness import run_trace, summarize
from bench.workload import make_workload
from tokquay.baseline import StaticBatchEngine
from tokquay.engine import Engine
from tokquay.kv_cache import BlockAllocator
from tokquay.model import GPT2
from tokquay.scheduler import SchedulerConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path("bench/results/results.json")
MEMORY_FRACTION: float | None = None  # set from --memory-fraction in main()
PLACEMENT: dict = {}  # set from --cpu-affinity / --high-priority in main()
MAX_LEN = 512  # the workload's longest request: 256 prompt + 256 output tokens
DEFAULT_BUDGET_TOKENS = 16384  # 1024 blocks of 16: 1.125 GiB of KV
DEFAULT_MEMORY_FRACTION = 0.8


# ------------------------------------------------------------------------------ process placement
def pin_process(affinity_mask: int | None, high_priority: bool) -> dict:
    """Pin this process to the logical CPUs in ``affinity_mask`` and/or raise its priority.
    Returns what was applied (recorded in the results)."""
    applied: dict = {}
    if platform.system() == "Windows":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = k32.GetCurrentProcess()
        if affinity_mask and not k32.SetProcessAffinityMask(handle, affinity_mask):
            raise OSError(f"could not set CPU affinity {affinity_mask:#x}")
        if high_priority and not k32.SetPriorityClass(handle, 0x00000080):  # HIGH_PRIORITY_CLASS
            raise OSError("could not raise the process priority")
    else:
        if affinity_mask:
            os.sched_setaffinity(0, {i for i in range(64) if affinity_mask >> i & 1})
        if high_priority:
            os.nice(-10)  # needs privileges
    if affinity_mask:
        applied["cpu_affinity"] = f"{affinity_mask:#x}"
    if high_priority:
        applied["high_priority"] = True
    return applied


# ------------------------------------------------------------------------------ building systems
def build(system: str, model: GPT2, *, budget_tokens: int, block_size: int = 16, max_len: int = MAX_LEN,
          max_num_seqs: int = 64, max_num_batched_tokens: int = 2048, static_rows: int | None = None):  # fmt: skip
    """``(system object, KV bytes per token slot)`` with ``budget_tokens`` slots of KV memory."""
    if system == "tokquay":
        allocator = BlockAllocator(model.cfg, budget_tokens // block_size, block_size, device=DEVICE)
        config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)
        return Engine(model, allocator, config), allocator.bytes_per_block // block_size
    if system == "static":
        engine = StaticBatchEngine(model, static_rows or budget_tokens // max_len, max_len, device=DEVICE)
        return engine, engine.bytes_per_token
    raise ValueError(system)


def free_gpu() -> None:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def warm_up(model: GPT2, **cfg) -> None:
    """Untimed runs so CUDA kernels, allocator caches and cuBLAS heuristics are warm."""
    workload = make_workload(math.inf, 12, seed=999, prompt_len=(16, 128), output_len=(8, 16))
    for system in ("tokquay", "static"):
        obj, _ = build(system, model, **cfg)
        run_trace(obj, workload)
        del obj
    free_gpu()


def gpu_health(retries_before: int) -> dict:
    """Allocator sanity numbers for the run that just finished (empty on CPU)."""
    if DEVICE != "cuda":
        return {}
    total = torch.cuda.get_device_properties(0).total_memory
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "total_mib": total / 2**20,
        "alloc_retries": torch.cuda.memory_stats()["num_alloc_retries"] - retries_before,
    }


def run_one(model: GPT2, system: str, workload, **cfg) -> dict:
    obj, bytes_per_token = build(system, model, **cfg)
    free_gpu()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    retries_before = torch.cuda.memory_stats()["num_alloc_retries"] if DEVICE == "cuda" else 0
    result = run_trace(obj, workload)
    summary = summarize(result, bytes_per_token)
    summary["gpu"] = gpu_health(retries_before)
    if summary["gpu"] and summary["gpu"]["peak_reserved_mib"] > 0.95 * summary["gpu"]["total_mib"]:
        print("  WARNING: the allocator reserved nearly all of VRAM; step times in this run may be inflated by memory spill", flush=True)
    summary["config"] = {"system": system, **cfg, "kv_reserved_mib": cfg["budget_tokens"] * bytes_per_token / 2**20}
    if system == "static":
        summary["config"]["rows"] = obj.batch_size
        summary["stats"]["batches"] = obj.num_batches
    del obj
    free_gpu()
    return summary


def one_line(name: str, s: dict) -> str:
    st = s["stats"]
    return (
        f"{name:34s} {s['throughput_tok_s']:7.1f} tok/s | TTFT p50/p99 {s['ttft_ms']['p50']:8.0f}/{s['ttft_ms']['p99']:8.0f} ms"
        f" | ITL p50/p99 {s['itl_ms']['p50']:6.1f}/{s['itl_ms']['p99']:7.1f} ms | peak KV {s['peak_kv_allocated_mib']:6.0f} MiB"
        f" | max batch {s['max_concurrent']:3d} | preempt {st.get('num_preemptions', 0)} | {s['duration_s']:.0f}s"
        + (f" | reserved {s['gpu']['peak_reserved_mib']:.0f} MiB, {s['gpu']['alloc_retries']} retries" if s.get("gpu") else "")
    )


# ------------------------------------------------------------------------------ results file
class Store:
    def __init__(self, path: Path = RESULTS_PATH, update_meta: bool = True):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"meta": {}, "runs": {}}
        if update_meta:  # reading results for a report must not replace what the runs recorded
            self.data["meta"] = {
                "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else platform.processor(),
                "torch": torch.__version__,
                "dtype": "float32",
                "max_len": MAX_LEN,
                "memory_fraction": MEMORY_FRACTION,
                **PLACEMENT,
                "date": time.strftime("%Y-%m-%d"),
            }

    def add(self, key: str, run: dict) -> None:
        self.data["runs"][key] = run
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


# ------------------------------------------------------------------------------ experiments
def cmd_calibrate(args, model, store):
    """Everything arrives at once: the most work each system can be given, so its saturated throughput."""
    workload = make_workload(math.inf, args.n, seed=args.seed)
    mean_out = sum(r.output_len for r in workload) / len(workload)
    for system in ("tokquay", "static"):
        s = run_one(model, system, workload, budget_tokens=args.budget_tokens)
        store.add(f"calibrate/{system}", {"n": args.n, "seed": args.seed, "summary": s})
        print(one_line(f"calibrate {system}", s), flush=True)
        print(f"   -> about {s['throughput_tok_s'] / mean_out:.2f} requests/s at {mean_out:.0f} output tokens each", flush=True)


def cmd_sweep(args, model, store):
    for rate in parse_floats(args.rates):
        workload = make_workload(rate, args.n, seed=args.seed)
        for system in args.systems.split(","):
            s = run_one(model, system, workload, budget_tokens=args.budget_tokens)
            store.add(f"sweep/{system}/rate={rate:g}{args.tag}", {"rate": rate, "n": args.n, "seed": args.seed, "summary": s})
            print(one_line(f"sweep {system} @ {rate:g} req/s", s), flush=True)


def cmd_blocksize(args, model, store):
    for rate in parse_floats(args.rates):
        workload = make_workload(rate, args.n, seed=args.seed)
        for block_size in (16, 32):
            s = run_one(model, "tokquay", workload, budget_tokens=args.budget_tokens, block_size=block_size)
            key = f"blocksize/bs={block_size}/rate={rate:g}/tokens={args.budget_tokens}"
            store.add(key, {"rate": rate, "n": args.n, "seed": args.seed, "summary": s})
            print(one_line(f"block {block_size} @ {rate:g} req/s, {args.budget_tokens} tok", s) + f" | KV util {s['kv_utilization']:.3f}", flush=True)


def cmd_pool(args, model, store):
    workload = make_workload(args.rate, args.n, seed=args.seed)
    for tokens in (int(x) for x in args.budgets.split(",")):
        for system in ("tokquay", "static"):
            s = run_one(model, system, workload, budget_tokens=tokens)
            store.add(f"pool/{system}/tokens={tokens}", {"rate": args.rate, "n": args.n, "seed": args.seed, "summary": s})
            print(one_line(f"pool {system} {tokens} tok", s), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rate=False):
        p.add_argument("--n", type=int, default=100, help="requests per run")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS, help="KV memory, in token slots")
        p.add_argument("--no-warmup", action="store_true")
        p.add_argument("--cpu-affinity", default=None, help="hex mask of logical CPUs to pin to, e.g. FF for the performance cores")
        p.add_argument("--high-priority", action="store_true", help="run at high process priority")
        p.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION,
                       help="cap the CUDA caching allocator at this fraction of VRAM (0 = uncapped)")  # fmt: skip
        if rate:
            p.add_argument("--rate", type=float, required=True, help="arrival rate, requests/s")

    common(sub.add_parser("calibrate"))
    p = sub.add_parser("sweep")
    common(p)
    p.add_argument("--rates", required=True, help="comma-separated arrival rates, requests/s")
    p.add_argument("--systems", default="tokquay,static")
    p.add_argument("--tag", default="", help='suffix for the result key, e.g. "#repeat" to keep a second run of the same config')
    p = sub.add_parser("blocksize")
    common(p)
    p.add_argument("--rates", required=True)
    p = sub.add_parser("pool")
    common(p, rate=True)
    p.add_argument("--budgets", default="16384,8192,4096,2048", help="comma-separated KV budgets in token slots")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    if args.cmd == "report":
        from bench.report import report

        return report(Store(update_meta=False), Path("Documentation"))

    global MEMORY_FRACTION, PLACEMENT
    PLACEMENT = pin_process(int(args.cpu_affinity, 16) if args.cpu_affinity else None, args.high_priority)
    MEMORY_FRACTION = args.memory_fraction or None
    if DEVICE == "cuda" and MEMORY_FRACTION:
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    model = GPT2.from_pretrained("gpt2").to(DEVICE).eval()
    if not args.no_warmup:
        warm_up(model, budget_tokens=args.budget_tokens)
    store = Store()
    print(f"[{store.data['meta']['gpu']}] budget {args.budget_tokens} token slots, {args.n} requests per run", flush=True)
    {"calibrate": cmd_calibrate, "sweep": cmd_sweep, "blocksize": cmd_blocksize, "pool": cmd_pool}[args.cmd](args, model, store)


if __name__ == "__main__":
    main()
