# tests/test_flash_v1_ref.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
import torch
from attention.naive import sdpa_naive
from attention.flash_v1_ref import flashattn_v1_forward_ref
 
 
def _make_qkv(device, dtype, B=2, H=4, L=256, D=64):
    torch.manual_seed(0)
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)
    return q, k, v
 
 
def test_flash_v1_ref_matches_naive_noncausal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    q, k, v = _make_qkv(device, dtype)
 
    out_ref = sdpa_naive(q, k, v, is_causal=False, dropout_p=0.0)
    out_fa  = flashattn_v1_forward_ref(q, k, v, is_causal=False, block_q=128, block_kv=128)
 
    torch.testing.assert_close(out_fa, out_ref, rtol=1e-2, atol=1e-2)
 
 
def test_flash_v1_ref_matches_naive_causal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    q, k, v = _make_qkv(device, dtype)
 
    out_ref = sdpa_naive(q, k, v, is_causal=True, dropout_p=0.0)
    out_fa  = flashattn_v1_forward_ref(q, k, v, is_causal=True, block_q=128, block_kv=128)
 
    torch.testing.assert_close(out_fa, out_ref, rtol=1e-2, atol=1e-2)