# bench/bench_flash_paper.py
import os, csv, math, argparse
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from attention.naive import standard_attention
from attention.flash_attn_paper import flashattention_paper

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _time_cuda_events(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _sync()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters

def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)

def make_key_padding_mask(B, L, pad_frac, device):
    # True means masked (padding) in your flashattention code
    if pad_frac <= 0:
        return None
    valid = int(round(L * (1.0 - pad_frac)))
    mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    mask[:, valid:] = True
    return mask

def make_attn_mask_from_kpm(kpm):
    # Convert (B,L) padding mask to additive mask broadcastable to (B,1,1,L)
    # torch SDPA accepts attn_mask; bool or additive. We'll use additive (-inf).
    if kpm is None:
        return None
    B, L = kpm.shape
    attn_bias = torch.zeros((B, 1, 1, L), device=kpm.device)
    attn_bias = attn_bias.masked_fill(kpm[:, None, None, :], float("-inf"))
    return attn_bias

@dataclass
class Cfg:
    B: int = 2
    H: int = 8
    D: int = 64
    dtype: str = "fp16"
    warmup: int = 20
    iters: int = 50

def dtype_from_str(s):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[s]

def run_one(impl, q, k, v, causal, kpm, dropout_p, block_q, block_k):
    if impl == "standard":
        return standard_attention(q, k, v, causal=causal, key_padding_mask=kpm)
    elif impl == "flash_paper":
        return flashattention_paper(q, k, v, causal=causal, key_padding_mask=kpm,
                                   dropout_p=dropout_p, block_q=block_q, block_k=block_k)
    elif impl == "torch_sdpa":
        # For padding: pass attn_mask; for causal: use is_causal
        attn_mask = make_attn_mask_from_kpm(kpm)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=dropout_p, is_causal=causal)
    else:
        raise ValueError(impl)

def bench_case(L, cfg: Cfg, mask_case, dropout_p, pad_frac, block_q, block_k, out_rows):
    device = "cuda"
    dtype = dtype_from_str(cfg.dtype)

    q = torch.randn(cfg.B, cfg.H, L, cfg.D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(cfg.B, cfg.H, L, cfg.D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(cfg.B, cfg.H, L, cfg.D, device=device, dtype=dtype, requires_grad=True)

    causal = mask_case in ("causal", "causal+padding")
    kpm = make_key_padding_mask(cfg.B, L, pad_frac if "padding" in mask_case else 0.0, device)

    impls = ["standard", "flash_paper", "torch_sdpa"]

    for impl in impls:
        # warmup
        for _ in range(cfg.warmup):
            out = run_one(impl, q, k, v, causal, kpm, dropout_p, block_q, block_k)
            loss = out.float().square().mean()
            loss.backward()
            q.grad = None; k.grad = None; v.grad = None
        _sync()

        # forward timing
        torch.cuda.reset_peak_memory_stats()
        def fwd():
            run_one(impl, q, k, v, causal, kpm, dropout_p, block_q, block_k)
        fwd_ms = _time_cuda_events(fwd, cfg.iters)
        peak_fwd = _peak_mb()

        # fwd+bwd timing
        torch.cuda.reset_peak_memory_stats()
        def fwd_bwd():
            out = run_one(impl, q, k, v, causal, kpm, dropout_p, block_q, block_k)
            loss = out.float().square().mean()
            loss.backward()
            q.grad = None; k.grad = None; v.grad = None
        fwd_bwd_ms = _time_cuda_events(fwd_bwd, cfg.iters)
        peak_fwd_bwd = _peak_mb()

        out_rows.append({
            "gpu": torch.cuda.get_device_name(0),
            "impl": impl,
            "mask_case": mask_case,
            "dropout_p": dropout_p,
            "pad_frac": pad_frac if "padding" in mask_case else 0.0,
            "causal": causal,
            "L": L,
            "B": cfg.B, "H": cfg.H, "D": cfg.D,
            "dtype": cfg.dtype,
            "block_q": block_q, "block_k": block_k,
            "fwd_ms": fwd_ms,
            "fwd_bwd_ms": fwd_bwd_ms,
            "peak_fwd_mb": peak_fwd,
            "peak_fwd_bwd_mb": peak_fwd_bwd,
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,256,512,1024,2048,4096")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    ap.add_argument("--pad-frac", type=float, default=0.25)  # 25% padded tokens
    ap.add_argument("--dropouts", default="0.0,0.1")
    ap.add_argument("--block-q", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=128)
    ap.add_argument("--out", default="results/flash_paper_bench.csv")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = Cfg(dtype=args.dtype, iters=args.iters, warmup=args.warmup)
    L_list = [int(x) for x in args.lengths.split(",") if x.strip()]
    dropouts = [float(x) for x in args.dropouts.split(",") if x.strip()]

    rows = []
    mask_cases = ["none", "causal", "padding", "causal+padding"]

    print("GPU:", torch.cuda.get_device_name(0))
    for dropout_p in dropouts:
        for mask_case in mask_cases:
            for L in L_list:
                try:
                    bench_case(L, cfg, mask_case, dropout_p, args.pad_frac, args.block_q, args.block_k, rows)
                    print(f"OK: dropout={dropout_p} mask={mask_case} L={L}")
                except RuntimeError as e:
                    print(f"SKIP (OOM?): dropout={dropout_p} mask={mask_case} L={L} -> {e}")
                    torch.cuda.empty_cache()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()