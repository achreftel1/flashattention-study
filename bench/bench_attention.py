# bench/bench_attention.py
import os
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional

import torch
import torch.nn.functional as F

# --- implementations (adjust these imports to match repo) ---
from attention.naive import sdpa_naive, standard_attention, mha_forward  # implementations
from attention.attn_baseline import scaled_dot_product_attention  # baseline


# -------------------------
# Config + helpers
# -------------------------
@dataclass
class BenchConfig:
    B: int = 2
    H: int = 8
    D: int = 64
    causal: bool = False
    dtype: str = "fp16"      # "fp16" | "bf16" | "fp32"
    device: str = "cuda"
    warmup: int = 20
    iters: int = 50
    backward: bool = True    # measure fwd+bwd when True; still also measures fwd-only


def _get_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype_str not in m:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(m.keys())}.")
    return m[dtype_str]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _fwd_only(fn: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    out = fn(q, k, v)
    # Some of your functions return (out, extra). Normalize here.
    if isinstance(out, tuple):
        out = out[0]
    return out


def _fwd_bwd(fn: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    out = fn(q, k, v)
    if isinstance(out, tuple):
        out = out[0]
    # simple scalar loss to force backward
    loss = out.float().square().mean()
    loss.backward()
    return out


def _reset_grads(*tensors: torch.Tensor):
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def _time_cuda_events(loop_body: Callable[[], None], iters: int) -> float:
    """
    Returns average milliseconds per iteration using CUDA events.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    _sync()
    start.record()
    for _ in range(iters):
        loop_body()
    end.record()
    _sync()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


def _peak_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# -------------------------
# Benchmarks
# -------------------------
def make_impls(causal: bool) -> List[Dict[str, Any]]:
    """
    Return a list of implementations to benchmark.
    Wrap each function so signature is fn(q,k,v) -> out or (out, aux).
    """
    impls = []

    # 1) Your baseline
    def baseline(q, k, v):
        # adjust if your baseline returns (out, lse) or similar
        return scaled_dot_product_attention(q, k, v, causal=causal)

    impls.append({"name": "baseline_scaled_dot_product_attention", "kind": "qkv", "fn": baseline})

    # 2) naive
    def naive(q, k, v):
        return sdpa_naive(q, k, v, is_causal=causal)

    impls.append({"name": "naive_sdpa", "kind": "qkv", "fn": naive})

    # 3) your "standard_attention"
    def standard(q, k, v):
        return standard_attention(q, k, v, causal=causal)

    impls.append({"name": "standard_attention", "kind": "qkv", "fn": standard})

    # 4) PyTorch SDPA (reference)
    def torch_sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    impls.append({"name": "torch_sdpa", "kind": "qkv", "fn": torch_sdpa})

    # 5) MHA forward (if implemented) end-to-end
    def mha(x, wq, wk, wv, wo, nheads):
        return mha_forward(x, wq, wk, wv, wo, nheads=nheads, causal=causal)
    impls.append({"name": "mha_forward_end2end", "kind": "mha", "fn": mha})


    return impls


def benchmark_one(L: int, cfg: BenchConfig, impl: Dict[str, Any]) -> Dict[str, Any]:
    assert cfg.device == "cuda", "This harness is intended for CUDA benchmarking."
    assert torch.cuda.is_available(), "CUDA not available. Install CUDA-enabled PyTorch and drivers."

    dtype = _get_dtype(cfg.dtype)
    fn = impl["fn"]
    kind = impl.get("kind", "qkv")

    
    # -------------------------
    # Input creation
    # -------------------------
    if kind == "qkv":

        # (B,H,L,D)
        q = torch.randn(cfg.B, cfg.H, L, cfg.D, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)
        k = torch.randn(cfg.B, cfg.H, L, cfg.D, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)
        v = torch.randn(cfg.B, cfg.H, L, cfg.D, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)

        @torch.no_grad()
        def fwd_call():
            out = fn(q, k, v)
            return out[0] if isinstance(out, tuple) else out

        def bwd_call():
            out = fn(q, k, v)
            out = out[0] if isinstance(out, tuple) else out
            loss = out.float().square().mean()
            loss.backward()
            q.grad = None; k.grad = None; v.grad = None
        
    elif kind == "mha":
            
        # MHA expects x: (B,L,E) and weights (E,E)
        E = cfg.H * cfg.D
        x = torch.randn(cfg.B, L, E, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)

        # Weights: (E,E). For training-like backward, give them grads too.
        # If want inference-like (fixed weights), set requires_grad=False for weights.
        wq = torch.randn(E, E, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)
        wk = torch.randn(E, E, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)
        wv = torch.randn(E, E, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)
        wo = torch.randn(E, E, device=cfg.device, dtype=dtype, requires_grad=cfg.backward)

        @torch.no_grad()
        def fwd_call():
            return fn(x, wq, wk, wv, wo, cfg.H)

        def bwd_call():
            out = fn(x, wq, wk, wv, wo, cfg.H)
            loss = out.float().square().mean()
            loss.backward()
            x.grad = None; wq.grad = None; wk.grad = None; wv.grad = None; wo.grad = None

    else:
        raise ValueError(f"Unknown impl kind: {kind}")

    # -------------------------
    # Warmup
    # -------------------------

    
    for _ in range(cfg.warmup):
        if cfg.backward:
            bwd_call()
        else:
            fwd_call()
    _sync()



    
    # -------------------------
    # Forward timing + mem
    # -------------------------

    torch.cuda.reset_peak_memory_stats()
    fwd_ms = _time_cuda_events(fwd_call, cfg.iters)
    peak_fwd_mb = _peak_mem_mb()

    fwd_bwd_ms = None
    peak_fwd_bwd_mb = None
    if cfg.backward:
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd_ms = _time_cuda_events(bwd_call, cfg.iters)
        peak_fwd_bwd_mb = _peak_mem_mb()

    row = {
        "impl": impl["name"],
        "L": L,
        "B": cfg.B,
        "H": cfg.H,
        "D": cfg.D,
        "E": cfg.H * cfg.D,                # helpful to see embedding dim for MHA
        "dtype": cfg.dtype,
        "causal": cfg.causal,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "fwd_ms": float(fwd_ms),
        "peak_mem_fwd_mb": float(peak_fwd_mb),
        "fwd_bwd_ms": None if fwd_bwd_ms is None else float(fwd_bwd_ms),
        "peak_mem_fwd_bwd_mb": None if peak_fwd_bwd_mb is None else float(peak_fwd_bwd_mb),
        "gpu": torch.cuda.get_device_name(0),
    }
    return row


def run_bench(cfg: BenchConfig, L_list: List[int], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    impls = make_impls(cfg.causal)
    rows: List[Dict[str, Any]] = []

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {cfg}")
    print(f"Lengths: {L_list}\n")

    for L in L_list:
        for impl in impls:
            try:
                r = benchmark_one(L, cfg, impl)
                rows.append(r)
                print(
                    f"[OK] {impl['name']:35s} L={L:5d} "
                    f"fwd={r['fwd_ms']:.3f}ms "
                    f"peak_fwd={r['peak_mem_fwd_mb']:.1f}MB "
                    + (f"fwd+bwd={r['fwd_bwd_ms']:.3f}ms peak_fwd+bwd={r['peak_mem_fwd_bwd_mb']:.1f}MB"
                       if r["fwd_bwd_ms"] is not None else "")
                )
            except RuntimeError as e:
                # Often OOM for large L; clear cache and continue
                print(f"[SKIP] {impl['name']} L={L} : {type(e).__name__} - {e}")
                torch.cuda.empty_cache()

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        print(f"\nSaved CSV -> {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--lengths", type=str, default="256,512,1024,2048,4096,8192")
    p.add_argument("--out", type=str, default="bench/results_standard_attention.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = BenchConfig(
        B=args.B,
        H=args.H,
        D=args.D,
        causal=args.causal,
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        backward=not args.no_backward,
        device="cuda",
    )

    L_list = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    run_bench(cfg, L_list, args.out)