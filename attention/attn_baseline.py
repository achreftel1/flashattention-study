# src/attn_baseline.py
import math
import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Baseline scaled dot-product attention.

    Shapes:
      q: (B, H, L, D)
      k: (B, H, S, D)
      v: (B, H, S, D)
      attn_mask: broadcastable to (B, H, L, S) with True for ALLOW or False for BLOCK
      key_padding_mask: (B, S) with True for PAD (to be masked out)
      returns:
        out: (B, H, L, D)
        p:   (B, H, L, S) attention probabilities
    """
    B, H, L, D = q.shape
    _, _, S, _ = k.shape

    scale = 1.0 / math.sqrt(D)  # from Transformer scaled dot-product attention [1](https://arxiv.org/abs/1706.03762)[2](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, S)

    # Causal masking: disallow j > i
    if causal:
        causal_mask = torch.ones((L, S), device=scores.device, dtype=torch.bool).tril()
        # causal_mask True means keep, False means block
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    # Generic attention mask
    if attn_mask is not None:
        # Convention: attn_mask True=keep, False=block
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    # Key padding mask: True means pad -> block it
    if key_padding_mask is not None:
        # shape (B, 1, 1, S) broadcast across heads and query positions
        kpm = key_padding_mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(kpm, float("-inf"))

    p = torch.softmax(scores, dim=-1)  # (B, H, L, S)
    out = torch.matmul(p, v)           # (B, H, L, D)
    return out, p


def mha_forward(
    x: Tensor,
    wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
    nheads: int,
    attn_mask: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> Tensor:
    """
    Minimal multi-head attention forward pass (no bias, no dropout).
    Shapes:
      x:  (B, L, E)
      w*: (E, E)
    Returns:
      y:  (B, L, E)
    """
    B, L, E = x.shape
    assert E % nheads == 0
    D = E // nheads

    q = x @ wq
    k = x @ wk
    v = x @ wv

    # reshape to (B, H, L, D)
    q = q.view(B, L, nheads, D).transpose(1, 2)
    k = k.view(B, L, nheads, D).transpose(1, 2)
    v = v.view(B, L, nheads, D).transpose(1, 2)

    out, _ = scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        causal=causal,
    )

    # back to (B, L, E)
    out = out.transpose(1, 2).contiguous().view(B, L, E)
    y = out @ wo
    return y