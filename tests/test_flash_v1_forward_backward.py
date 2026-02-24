import torch
from attention.naive import standard_attention
from attention.flash_attn_paper import flashattention_paper

def test_flashattention_forward_matches_baseline():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    B,H,L,D = 2,8,256,64

    q = torch.randn(B,H,L,D, device=device, dtype=dtype)
    k = torch.randn(B,H,L,D, device=device, dtype=dtype)
    v = torch.randn(B,H,L,D, device=device, dtype=dtype)

    out_ref = standard_attention(q,k,v, causal=True)
    out_fa = flashattention_paper(q,k,v, causal=True, dropout_p=0.0, block_q=128, block_k=128)

    torch.testing.assert_close(out_fa, out_ref, atol=2e-2 if dtype==torch.float16 else 1e-4,
                               rtol=2e-2 if dtype==torch.float16 else 1e-4)
    

def test_flashattention_backward_matches_baseline():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    B,H,L,D = 2,8,128,64

    q1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)
    v1 = torch.randn(B,H,L,D, device=device, dtype=dtype, requires_grad=True)

    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)

    out_ref = standard_attention(q1,k1,v1, causal=True)
    loss_ref = out_ref.float().square().mean()
    loss_ref.backward()

    out_fa = flashattention_paper(q2,k2,v2, causal=True, dropout_p=0.0)
    loss_fa = out_fa.float().square().mean()
    loss_fa.backward()

    atol = 5e-2 if dtype == torch.float16 else 5e-4
    rtol = 5e-2 if dtype == torch.float16 else 5e-4
    torch.testing.assert_close(q2.grad, q1.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k2.grad, k1.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v2.grad, v1.grad, atol=atol, rtol=rtol)