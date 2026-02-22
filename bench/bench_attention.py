# bench/bench_attention.py (or current bench file)
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


from attention.naive import sdpa_naive

from attention.attn_baseline import scaled_dot_product_attention
from attention.naive import standard_attention



def run_bench(label, fn, q, k, v):
    # Make sure GPU work is synchronized for consistent timing
    if q.is_cuda:
        torch.cuda.synchronize()

    t = Timer(
        stmt="fn(q,k,v)",
        globals={"fn": fn, "q": q, "k": k, "v": v},
        label=label,
    )
    m = t.blocked_autorange(min_run_time=1.0)

    if q.is_cuda:
        torch.cuda.synchronize()

    print(m)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, H, L, D = 2, 8, 1024, 64
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    causal = True

    run_bench("naive", lambda q,k,v: sdpa_naive(q,k,v, is_causal=causal), q,k,v)
    run_bench("standard_algo (explicit backward not used here)", lambda q,k,v: standard_attention(q,k,v, causal=causal), q,k,v)
    run_bench("pytorch_sdpa", lambda q,k,v: F.scaled_dot_product_attention(q,k,v, is_causal=causal), q,k,v)



if __name__ == "__main__":
    main()