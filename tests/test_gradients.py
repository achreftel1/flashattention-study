# tests/test_gradients.py
import torch
import pytest
from   attention.attn_baseline import scaled_dot_product_attention


@pytest.mark.parametrize("B,H,L,D", [(1, 2, 16, 8)])
def test_gradcheck(B, H, L, D):
    torch.manual_seed(0)
    device = "cpu"  # gradcheck on CPU
    dtype = torch.float64

    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

    def fn(q, k, v):
        out, _ = scaled_dot_product_attention(q, k, v, causal=True)
        return out.sum()

    torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-4)