import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
import torch
import torch.nn.functional as F
from attention.naive import sdpa_naive
 
 
def test_sdpa_matches_pytorch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
 
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 256, 64
 
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)
 
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    out_my = sdpa_naive(q, k, v, is_causal=True, dropout_p=0.0)
 
    torch.testing.assert_close(out_my, out_ref, rtol=1e-2, atol=1e-2)