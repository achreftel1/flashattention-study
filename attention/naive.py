import math
import torch

def sdpa_naive(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    q, k, v: [B, H, L, D]
    Implements: softmax((QK^T)/sqrt(D) + mask) @ V
    """

    d = q.size(-1)
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # causal mask
    if is_causal:
        Lq, Lk = q.size(-2), k.size(-2)
        causal = torch.ones((Lq, Lk), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    # optional mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn = torch.dropout(attn, dropout_p, train=True)

    out = torch.matmul(attn, v)
    return out