import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import pytest
import torch.nn.functional as F

from attention.naive import sdpa_naive
from attention.attn_baseline import scaled_dot_product_attention


def torch_sdpa(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)


def run_impl(name, q, k, v, causal):
    if name == "naive":
        return sdpa_naive(q, k, v, is_causal=causal)
    elif name == "baseline":
        out, _ = scaled_dot_product_attention(q, k, v, causal=causal)
        return out
    else:
        raise ValueError(name)


@pytest.mark.parametrize("impl", ["naive", "baseline"])
@pytest.mark.parametrize("B,H,L,D", [(2, 4, 64, 32), (1, 2, 128, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_torch_sdpa(impl, B, H, L, D, causal):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    out = run_impl(impl, q, k, v, causal=causal)
    out_ref = torch_sdpa(q, k, v, causal=causal)

    # Looser tolerances for fp16
    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)