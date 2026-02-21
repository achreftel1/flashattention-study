import torch
import pytest
from attention.attn_baseline import scaled_dot_product_attention
from attention.flash_v1 import flash_attn_v1, flash_attn_fwd_blockwise


@pytest.mark.parametrize("B,H,L,D,block_k", [(2, 4, 128, 32, 64), (1, 2, 256, 64, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_v1_forward_matches_baseline(B,H,L,D,block_k, causal):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    q = torch.randn(B,H,L,D, device=device, dtype=dtype)
    k = torch.randn(B,H,L,D, device=device, dtype=dtype)
    v = torch.randn(B,H,L,D, device=device, dtype=dtype)

    out_ref, _ = scaled_dot_product_attention(q, k, v, causal=causal)
    out_fa, lse = flash_attn_fwd_blockwise(q, k, v, block_k=block_k, causal=causal)

    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out_fa, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("B,H,L,D,block_k", [(2, 4, 128, 32, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_v1_backward_matches_baseline(B,H,L,D,block_k, causal):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    q1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    v1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)

    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)

    # Baseline
    out_ref, _ = scaled_dot_product_attention(q1, k1, v1, causal=causal)
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    # FlashAttention v1
    out_fa = flash_attn_v1(q2, k2, v2, block_k=block_k, causal=causal)
    loss_fa = out_fa.square().mean()
    loss_fa.backward()

    atol = 5e-2 if dtype == torch.float16 else 5e-4
    rtol = 5e-2 if dtype == torch.float16 else 5e-4

    torch.testing.assert_close(q2.grad, q1.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k2.grad, k1.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v2.grad, v1.grad, atol=atol, rtol=rtol)

def test_flash_v1_gradcheck_cpu():
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float64

    B,H,L,D = 1, 2, 16, 8
    q = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)

    def fn(q,k,v):
        out = flash_attn_v1(q,k,v, block_k=8, causal=True)
        return out.sum()

    torch.autograd.gradcheck(fn, (q,k,v), eps=1e-6, atol=1e-4, rtol=1e-4)