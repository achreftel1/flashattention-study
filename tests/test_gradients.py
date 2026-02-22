# tests/test_gradients.py **
import torch
import pytest
from attention.naive import standard_attention


@pytest.mark.parametrize("B,H,L,D", [(1, 2, 8, 4)])
@pytest.mark.parametrize("causal", [True, False])
def test_gradcheck_standard_attention(B, H, L, D, causal):
    torch.manual_seed(0)
    device = "cpu"          # gradcheck must be CPU
    dtype = torch.float64   # gradcheck uses double

    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

    def fn(q, k, v):
        out = standard_attention(q, k, v, causal=causal)
        return out.sum()

    torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-4)