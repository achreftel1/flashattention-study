import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
from attention.naive import sdpa_naive

def run_bench(label, fn, q, k, v):
    t = Timer(
        stmt="fn(q,k,v)",
        globals={"fn": fn, "q": q, "k": k, "v": v},
        label=label,
    )
    m = t.blocked_autorange(min_run_time=1.0)
    print(m)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, H, L, D = 2, 8, 1024, 64
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    run_bench("naive", lambda q,k,v: sdpa_naive(q,k,v, is_causal=True), q,k,v)
    run_bench("pytorch_sdpa", lambda q,k,v: F.scaled_dot_product_attention(q,k,v, is_causal=True), q,k,v)

if __name__ == "__main__":
    main()