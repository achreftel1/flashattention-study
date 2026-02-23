# attention/naive.py
import math
import torch
from torch import Tensor

import math
import torch

def sdpa_naive(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    q, k, v: [B, H, L, D] (self-attention assumes same L)
    Implements: softmax((QK^T)/sqrt(D) + mask) @ V
    """
    d = q.size(-1)
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,L)

    if is_causal:
        Lq, Lk = q.size(-2), k.size(-2)
        causal = torch.ones((Lq, Lk), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

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

def _build_allowed_mask(
    B: int, H: int, L: int, S: int,
    device,
    attn_mask: Tensor | None,
    key_padding_mask: Tensor | None,
    causal: bool,
) -> Tensor:
    """
    Returns allowed mask with shape (B, H, L, S), dtype=bool
    True = ALLOW, False = BLOCK
    """
    allowed = torch.ones((B, H, L, S), device=device, dtype=torch.bool)

    if causal:
        # allow j <= i
        causal_mask = torch.ones((L, S), device=device, dtype=torch.bool).tril()
        allowed &= causal_mask[None, None, :, :]  # broadcast

    if attn_mask is not None:
        # If boolean: True allow / False block
        if attn_mask.dtype == torch.bool:
            allowed &= attn_mask.to(device=device)
        else:
            # additive mask handled elsewhere, not in allowed mask
            pass

    if key_padding_mask is not None:
        # key_padding_mask: (B, S) with True for PAD => block those keys
        kpm = key_padding_mask.to(device=device, dtype=torch.bool)[:, None, None, :]  # (B,1,1,S)
        allowed &= ~kpm  # pad => block

    return allowed


class StandardAttentionFn(torch.autograd.Function):
    """
    Implements Standard Attention exactly like the paper algorithm:

    Forward (Algorithm 0):
      S = (Q K^T) / sqrt(D) (+ masks)
      P = softmax(S)
      O = P V

    Backward (Algorithm 3):
      dV = P^T dO
      dP = dO V^T
      dS_ij = P_ij (dP_ij - sum_l P_il dP_il)
      dQ = (dS K) / sqrt(D)
      dK = (dS^T Q) / sqrt(D)
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor, k: Tensor, v: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        causal: bool,
    ) -> Tensor:
        """
        q: (B, H, L, D)
        k: (B, H, S, D)
        v: (B, H, S, D)

        attn_mask:
          - bool mask broadcastable to (B,H,L,S): True allow, False block
          - OR additive mask broadcastable to (B,H,L,S): added to scores
        key_padding_mask: (B,S) True for PAD (blocked)
        causal: block j > i

        returns out: (B, H, L, D)
        """
        B, H, L, D = q.shape
        _, _, S, _ = k.shape
        device = q.device

        scale = 1.0 / math.sqrt(D)

        # Build boolean allowed mask (True keep / False block)
        allowed = _build_allowed_mask(B, H, L, S, device, attn_mask, key_padding_mask, causal)

        # Algorithm 0 - Step 1: S = QK^T / sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,S)

        # Additive mask (if provided and not bool)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            scores = scores + attn_mask.to(device=device)

        # Apply blocking: set blocked positions to -inf
        scores = scores.masked_fill(~allowed, float("-inf"))

        # Algorithm 0 - Step 2: P = softmax(S)
        # Stable softmax; assumes each row has at least one allowed key.
        # If you can have fully-masked rows, you must handle them explicitly.
        max_scores = torch.max(scores, dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)  # exp(-inf) -> 0
        exp_scores = exp_scores * allowed.to(exp_scores.dtype)  # ensure blocked are exactly 0
        denom = torch.sum(exp_scores, dim=-1, keepdim=True)
        p = exp_scores / denom  # (B,H,L,S)

        # Algorithm 0 - Step 3: O = P V
        out = torch.matmul(p, v)  # (B,H,L,D)

        # Save tensors for Algorithm 3 backward
        ctx.save_for_backward(q, k, v, p, allowed)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        """
        dout: (B,H,L,D)
        returns gradients for (q,k,v,attn_mask,key_padding_mask,causal)
        Only q,k,v have gradients; masks/causal are None.
        """
        q, k, v, p, allowed = ctx.saved_tensors
        scale = ctx.scale

        # Algorithm 3 - Step 1: dV = P^T dO
        dV = torch.matmul(p.transpose(-2, -1), dout)  # (B,H,S,D)

        # Algorithm 3 - Step 2: dP = dO V^T
        dP = torch.matmul(dout, v.transpose(-2, -1))  # (B,H,L,S)

        # Algorithm 3 - Step 3: dS = P ⊙ (dP - sum_l P_il dP_il)
        # row-wise dot: sum over keys (last dim)
        row_sum = torch.sum(dP * p, dim=-1, keepdim=True)  # (B,H,L,1)
        dS = p * (dP - row_sum)  # (B,H,L,S)

        # ensure masked positions stay zero
        dS = dS * allowed.to(dS.dtype)

        # Algorithm 3 - Step 4: dQ = dS K / sqrt(D)
        dQ = torch.matmul(dS, k) * scale  # (B,H,L,D)

        # Algorithm 3 - Step 5: dK = dS^T Q / sqrt(D)
        dK = torch.matmul(dS.transpose(-2, -1), q) * scale  # (B,H,S,D)

        # No grads for masks/causal flags
        return dQ, dK, dV, None, None, None


def standard_attention(
    q: Tensor, k: Tensor, v: Tensor,
    attn_mask: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> Tensor:
    """
    Canonical Standard Attention (Algorithm 0 + Algorithm 3)
    q: (B,H,L,D), k/v: (B,H,S,D) -> out: (B,H,L,D)
    """
    return StandardAttentionFn.apply(q, k, v, attn_mask, key_padding_mask, causal)


def mha_forward(
    x: Tensor,
    wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
    nheads: int,
    attn_mask: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> Tensor:
    """
    Minimal Multi-Head Attention forward built on the SAME standard_attention core.
    This is not a second algorithm—just projections + reshape around standard attention.

    x:  (B, L, E)
    w*: (E, E)
    returns y: (B, L, E)
    """
    B, L, E = x.shape
    assert E % nheads == 0
    D = E // nheads

    # Linear projections
    q = x @ wq
    k = x @ wk
    v = x @ wv

    # (B,L,E) -> (B,H,L,D)
    q = q.view(B, L, nheads, D).transpose(1, 2).contiguous()
    k = k.view(B, L, nheads, D).transpose(1, 2).contiguous()
    v = v.view(B, L, nheads, D).transpose(1, 2).contiguous()

    out = standard_attention(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, causal=causal)

    # (B,H,L,D) -> (B,L,E)
    out = out.transpose(1, 2).contiguous().view(B, L, E)
    y = out @ wo
    return y