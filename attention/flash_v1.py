# src/flash_v1.py
import math
import torch
from torch import Tensor


def _compute_dtype(x: Tensor) -> torch.dtype:
    # Use fp32 internally for fp16/bf16 stability; otherwise keep dtype.
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


def flash_attn_fwd_blockwise(
    q: Tensor, k: Tensor, v: Tensor,
    block_k: int = 128,
    causal: bool = False,
    key_padding_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    FlashAttention v1 conceptual forward (tiling + online softmax).

    Inputs:
      q, k, v: (B, H, L, D) / (B, H, S, D)
      key_padding_mask: (B, S) where True means PAD -> masked out

    Returns:
      out: (B, H, L, D)
      lse: (B, H, L)  # logsumexp per row of scores, used for backward recomputation
    """
    B, H, L, D = q.shape
    _, _, S, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    cdtype = _compute_dtype(q)
    q_ = q.to(cdtype)
    k_ = k.to(cdtype)
    v_ = v.to(cdtype)

    # Running max m and running sum l for each (B,H,L)
    m = torch.full((B, H, L), float("-inf"), device=q.device, dtype=cdtype)
    l = torch.zeros((B, H, L), device=q.device, dtype=cdtype)
    out = torch.zeros((B, H, L, D), device=q.device, dtype=cdtype)

    for start in range(0, S, block_k):
        end = min(start + block_k, S)
        kb = k_[:, :, start:end, :]    # (B,H,bs,D)
        vb = v_[:, :, start:end, :]    # (B,H,bs,D)

        scores = torch.matmul(q_, kb.transpose(-2, -1)) * scale  # (B,H,L,bs)

        # Key padding mask (True means pad -> mask out)
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, start:end].to(torch.bool)           # (B,bs)
            scores = scores.masked_fill(kpm[:, None, None, :], float("-inf"))

        # Causal mask for self-attn (typical). Works generally for L,S too:
        if causal:
            i = torch.arange(L, device=q.device)[:, None]                 # (L,1)
            j = torch.arange(start, end, device=q.device)[None, :]        # (1,bs)
            causal_mask = (j <= i)                                        # (L,bs)
            scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

        block_max = scores.max(dim=-1).values                             # (B,H,L)
        m_new = torch.maximum(m, block_max)                                # (B,H,L)

        # Rescale old accumulators by exp(m - m_new)
        exp_scale = torch.exp(m - m_new)                                   # (B,H,L)
        l = l * exp_scale
        out = out * exp_scale.unsqueeze(-1)

        # New block contributions
        p = torch.exp(scores - m_new.unsqueeze(-1))                        # (B,H,L,bs)
        l = l + p.sum(dim=-1)
        out = out + torch.matmul(p, vb)                                    # (B,H,L,D)

        m = m_new

    out = out / l.unsqueeze(-1)
    lse = m + torch.log(l)                                                 # logsumexp
    return out.to(dtype=q.dtype), lse.to(dtype=_compute_dtype(q))


class FlashAttnV1Fn(torch.autograd.Function):
    """
    FlashAttention v1 conceptual autograd:
      - forward computes output and lse (logsumexp)
      - backward recomputes probabilities blockwise from lse (no full attention matrix stored)
    This mirrors the paper’s “tiling + recomputation” principle. [1](https://arxiv.org/abs/2205.14135)[2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
    """

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor,
                block_k: int, causal: bool, key_padding_mask: Tensor | None):
        out, lse = flash_attn_fwd_blockwise(
            q, k, v,
            block_k=block_k,
            causal=causal,
            key_padding_mask=key_padding_mask,
        )
        # Save tensors needed for backward recomputation
        ctx.save_for_backward(q, k, v, out, lse, key_padding_mask if key_padding_mask is not None else torch.tensor([]))
        ctx.block_k = block_k
        ctx.causal = causal
        ctx.has_kpm = key_padding_mask is not None
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        q, k, v, out, lse, kpm_saved = ctx.saved_tensors
        block_k = ctx.block_k
        causal = ctx.causal
        key_padding_mask = kpm_saved if ctx.has_kpm else None

        B, H, L, D = q.shape
        _, _, S, _ = k.shape
        scale = 1.0 / math.sqrt(D)

        cdtype = _compute_dtype(q)
        q_ = q.to(cdtype)
        k_ = k.to(cdtype)
        v_ = v.to(cdtype)
        out_ = out.to(cdtype)
        dout_ = dout.to(cdtype)
        lse_ = lse.to(cdtype)   # (B,H,L)

        # delta = sum_j P_ij dP_ij = dout_i · out_i
        # This avoids a separate accumulation pass and fits the recomputation idea. [1](https://arxiv.org/abs/2205.14135)[2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
        delta = (dout_ * out_).sum(dim=-1)  # (B,H,L)

        dQ = torch.zeros_like(q_, dtype=cdtype)
        dK = torch.zeros_like(k_, dtype=cdtype)
        dV = torch.zeros_like(v_, dtype=cdtype)

        for start in range(0, S, block_k):
            end = min(start + block_k, S)
            kb = k_[:, :, start:end, :]  # (B,H,bs,D)
            vb = v_[:, :, start:end, :]  # (B,H,bs,D)

            scores = torch.matmul(q_, kb.transpose(-2, -1)) * scale  # (B,H,L,bs)

            if key_padding_mask is not None and key_padding_mask.numel() > 0:
                kpm = key_padding_mask[:, start:end].to(torch.bool)
                scores = scores.masked_fill(kpm[:, None, None, :], float("-inf"))

            if causal:
                i = torch.arange(L, device=q.device)[:, None]
                j = torch.arange(start, end, device=q.device)[None, :]
                causal_mask = (j <= i)
                scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

            # Recompute probabilities for this block:
            # P_block = exp(scores - lse_row)
            p = torch.exp(scores - lse_.unsqueeze(-1))  # (B,H,L,bs)

            # dV_block = P_block^T @ dout
            dV[:, :, start:end, :] += torch.matmul(p.transpose(-2, -1), dout_)  # (B,H,bs,D)

            # dP_block = dout @ V_block^T
            dP = torch.matmul(dout_, vb.transpose(-2, -1))  # (B,H,L,bs)

            # dS_block = P_block * (dP_block - delta)
            dS = p * (dP - delta.unsqueeze(-1))  # (B,H,L,bs)

            # Propagate through scores = (QK^T)*scale
            dQ += torch.matmul(dS, kb) * scale
            dK[:, :, start:end, :] += torch.matmul(dS.transpose(-2, -1), q_) * scale

        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None, None, None


def flash_attn_v1(q: Tensor, k: Tensor, v: Tensor,
                  block_k: int = 128,
                  causal: bool = False,
                  key_padding_mask: Tensor | None = None) -> Tensor:
    """
    User-facing wrapper.
    """
    return FlashAttnV1Fn.apply(q, k, v, block_k, causal, key_padding_mask)