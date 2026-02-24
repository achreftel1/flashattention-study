# attention/flash_attn_paper.py

"""
FlashAttention v1 (paper-aligned) implementation in pure PyTorch.

Implements:
- Algorithm 2: FlashAttention Forward Pass (tiling + online softmax)
- Algorithm 4: FlashAttention Backward Pass (recompute P from saved row stats)

Shapes:
    q, k, v: (B, H, L, D)  (self-attention, S=L)
Optional:
    key_padding_mask: (B, L) boolean, True means "masked out" (padding)

Notes:
- This is an *algorithmic* reproduction of FlashAttention v1 (exact), not the CUDA-optimized kernel.
- We accumulate in float32 for numerical stability.
- Dropout is optional; we save RNG state in forward and replay it in backward.
"""


from __future__ import annotations
import math
import torch
from torch import Tensor

def _to_acc_dtype(x: Tensor) -> Tensor:
    # accumulate in fp32 for fp16/bf16
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x

def _save_rng_state_for_device(device: torch.device):
    # Save CPU RNG state always; save CUDA RNG state for the specific device if CUDA.
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return cpu_state, cuda_state

def _restore_rng_state_for_device(device: torch.device, cpu_state, cuda_state):
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)

def _apply_causal_mask(scores: Tensor, q_start: int, k_start: int) -> Tensor:
    """
    scores: (B,H,Br,Bc)
    causal mask: key_index <= query_index
    """
    Br = scores.shape[-2]
    Bc = scores.shape[-1]
    device = scores.device
    q_idx = torch.arange(q_start, q_start + Br, device=device)[:, None]  # (Br,1)
    k_idx = torch.arange(k_start, k_start + Bc, device=device)[None, :]  # (1,Bc)
    mask = (k_idx <= q_idx)  # (Br,Bc)
    return scores.masked_fill(~mask[None, None, :, :], float("-inf"))

def _apply_key_padding_mask(scores: Tensor, key_padding_mask: Tensor, k_start: int) -> Tensor:
    """
    key_padding_mask: (B,L) True means masked (padding)
    scores: (B,H,Br,Bc)
    """
    Bc = scores.shape[-1]
    kpm_block = key_padding_mask[:, k_start:k_start + Bc].to(torch.bool)  # (B,Bc)
    return scores.masked_fill(kpm_block[:, None, None, :], float("-inf"))


def flashattention_forward_paper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    block_q: int = 128,
    block_k: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Paper-aligned forward (Algorithm 2):
    returns:
        out: (B,H,L,D)
        m:   (B,H,L)  running max per row (final)
        l:   (B,H,L)  running sumexp per row (final)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q,k,v must be (B,H,L,D)"
    B, H, L, D = q.shape
    assert k.shape == (B, H, L, D) and v.shape == (B, H, L, D), "self-attention expects k,v same shape as q"
    assert dropout_p >= 0.0 and dropout_p < 1.0

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    device = q.device
    q_ = _to_acc_dtype(q)
    k_ = _to_acc_dtype(k)
    v_ = _to_acc_dtype(v)

    # Initialize O, l, m in "HBM" (here: tensors)
    out = torch.zeros((B, H, L, D), device=device, dtype=q_.dtype)
    l = torch.zeros((B, H, L), device=device, dtype=q_.dtype)
    m = torch.full((B, H, L), float("-inf"), device=device, dtype=q_.dtype)

    # Outer loop over K/V blocks (j loop in paper)
    for k_start in range(0, L, block_k):
        k_end = min(k_start + block_k, L)
        K_j = k_[:, :, k_start:k_end, :]  # (B,H,Bc,D)
        V_j = v_[:, :, k_start:k_end, :]  # (B,H,Bc,D)
        Bc = k_end - k_start

        # Inner loop over Q blocks (i loop in paper)
        for q_start in range(0, L, block_q):
            q_end = min(q_start + block_q, L)
            Br = q_end - q_start

            Q_i = q_[:, :, q_start:q_end, :]          # (B,H,Br,D)
            O_i = out[:, :, q_start:q_end, :]         # (B,H,Br,D)
            l_i = l[:, :, q_start:q_end]              # (B,H,Br)
            m_i = m[:, :, q_start:q_end]              # (B,H,Br)

            # S_ij = scale * Q_i K_j^T  -> (B,H,Br,Bc)
            scores = torch.matmul(Q_i, K_j.transpose(-2, -1)) * softmax_scale

            # mask(S_ij)
            if key_padding_mask is not None:
                scores = _apply_key_padding_mask(scores, key_padding_mask, k_start)
            if causal:
                scores = _apply_causal_mask(scores, q_start, k_start)

            # m_tilde = rowmax(scores), P_tilde = exp(scores - m_tilde), l_tilde = rowsum(P_tilde)
            m_tilde = scores.max(dim=-1).values                      # (B,H,Br)
            all_masked = torch.isneginf(m_tilde)  # (B,H,Br) rows where everything is -inf
            m_tilde_safe = torch.where(all_masked, torch.zeros_like(m_tilde), m_tilde)
            # This prevents (-inf) - (-inf) → NaN
            P_tilde = torch.exp(scores - m_tilde_safe.unsqueeze(-1))
            P_tilde = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(P_tilde), P_tilde)
            l_tilde = P_tilde.sum(dim=-1)
            

            # m_new, l_new (online softmax update)
            m_new = torch.maximum(m_i, m_tilde)
            # exp(m_i - m_new) * l_i  + exp(m_tilde - m_new) * l_tilde
            exp_mi = torch.exp(m_i - m_new)
            exp_mt = torch.exp(m_tilde - m_new)
            l_new = exp_mi * l_i + exp_mt * l_tilde

            # Dropout on P_tilde (paper applies dropout to P_tilde before combining)
            if dropout_p > 0.0:
                # Mask values: 1/(1-p) with prob (1-p), else 0
                # (same expectation as original)
                keep = (torch.rand_like(P_tilde) > dropout_p).to(P_tilde.dtype)
                keep = keep / (1.0 - dropout_p)
                P_tilde = P_tilde * keep

            # Update O_i:
            # O_i <- diag(l_new)^(-1) ( diag(l_i)*exp(m_i-m_new)*O_i + exp(m_tilde-m_new)*P_tilde@V_j )
            # Here diag(l_i)*... is per-row scaling:
            term_old = (l_i * exp_mi).unsqueeze(-1) * O_i             # (B,H,Br,D)
            term_new = exp_mt.unsqueeze(-1) * torch.matmul(P_tilde, V_j)  # (B,H,Br,D)
            O_i_new = (term_old + term_new) / l_new.unsqueeze(-1)

            # Write back to "HBM"
            out[:, :, q_start:q_end, :] = O_i_new
            l[:, :, q_start:q_end] = l_new
            m[:, :, q_start:q_end] = m_new

    return out.to(dtype=q.dtype), m, l

def flashattention_backward_paper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    m: Tensor,
    l: Tensor,
    dout: Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    block_q: int = 128,
    block_k: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Paper-aligned backward (Algorithm 4).
    Recomputes P_ij blockwise using saved (m,l).

    returns: dQ, dK, dV (all shape B,H,L,D)
    """
    B, H, L, D = q.shape
    assert k.shape == (B, H, L, D) and v.shape == (B, H, L, D)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    device = q.device
    q_ = _to_acc_dtype(q)
    k_ = _to_acc_dtype(k)
    v_ = _to_acc_dtype(v)
    out_ = _to_acc_dtype(out)
    dout_ = _to_acc_dtype(dout)
    m_ = _to_acc_dtype(m)
    l_ = _to_acc_dtype(l)

    dQ = torch.zeros_like(q_, dtype=q_.dtype)
    dK = torch.zeros_like(k_, dtype=k_.dtype)
    dV = torch.zeros_like(v_, dtype=v_.dtype)

    # D_i = rowsum(dO_i ◦ O_i)  (Algorithm 4 line 19)
    # shape: (B,H,L)
    delta = (dout_ * out_).sum(dim=-1)

    # Outer loop over K/V blocks (j)
    for k_start in range(0, L, block_k):
        k_end = min(k_start + block_k, L)
        K_j = k_[:, :, k_start:k_end, :]  # (B,H,Bc,D)
        V_j = v_[:, :, k_start:k_end, :]  # (B,H,Bc,D)
        Bc = k_end - k_start

        # Accumulate per-block dK_j, dV_j on SRAM in paper
        dK_j = torch.zeros((B, H, Bc, D), device=device, dtype=q_.dtype)
        dV_j = torch.zeros((B, H, Bc, D), device=device, dtype=q_.dtype)

        # Inner loop over Q blocks (i)
        for q_start in range(0, L, block_q):
            q_end = min(q_start + block_q, L)
            Br = q_end - q_start

            Q_i = q_[:, :, q_start:q_end, :]           # (B,H,Br,D)
            O_i = out_[:, :, q_start:q_end, :]         # (B,H,Br,D)
            dO_i = dout_[:, :, q_start:q_end, :]       # (B,H,Br,D)
            m_i = m_[:, :, q_start:q_end]              # (B,H,Br)
            l_i = l_[:, :, q_start:q_end]              # (B,H,Br)
            delta_i = delta[:, :, q_start:q_end]       # (B,H,Br)

            # Recompute scores S_ij
            scores = torch.matmul(Q_i, K_j.transpose(-2, -1)) * softmax_scale  # (B,H,Br,Bc)

            if key_padding_mask is not None:
                scores = _apply_key_padding_mask(scores, key_padding_mask, k_start)
            if causal:
                scores = _apply_causal_mask(scores, q_start, k_start)

            # Recompute P_ij using saved (m,l):
            # P = diag(l_i)^(-1) * exp(scores - m_i)
            # (Algorithm 4 line 13)
            P = torch.exp(scores - m_i.unsqueeze(-1)) / l_i.unsqueeze(-1)      # (B,H,Br,Bc)

            # Dropout mask Z_ij (Algorithm 4 line 14)
            if dropout_p > 0.0:
                keep = (torch.rand_like(P) > dropout_p).to(P.dtype)
                Z = keep / (1.0 - dropout_p)
            else:
                Z = None

            # P_dropped = P ◦ Z (Algorithm 4 line 15)
            P_dropped = P if Z is None else (P * Z)

            # dV_j += (P_dropped)^T dO_i  (line 16)
            dV_j += torch.matmul(P_dropped.transpose(-2, -1), dO_i)  # (B,H,Bc,D)

            # dP_dropped = dO_i V_j^T  (line 17)
            dP_dropped = torch.matmul(dO_i, V_j.transpose(-2, -1))   # (B,H,Br,Bc)

            # dP = dP_dropped ◦ Z  (line 18)
            dP = dP_dropped if Z is None else (dP_dropped * Z)

            # dS = P ◦ (dP - D_i)  (line 20)
            dS = P * (dP - delta_i.unsqueeze(-1))                     # (B,H,Br,Bc)

            # dQ_i += scale * dS K_j  (line 21)
            dQ[:, :, q_start:q_end, :] += torch.matmul(dS, K_j) * softmax_scale

            # dK_j += scale * dS^T Q_i  (line 22)
            dK_j += torch.matmul(dS.transpose(-2, -1), Q_i) * softmax_scale

        # Write dK_j, dV_j back to HBM (line 24)
        dK[:, :, k_start:k_end, :] += dK_j
        dV[:, :, k_start:k_end, :] += dV_j

    return dQ.to(dtype=q.dtype), dK.to(dtype=q.dtype), dV.to(dtype=q.dtype)


class FlashAttentionPaperV1(torch.autograd.Function):
    """
    torch.autograd.Function implementing FlashAttention v1 (paper Algorithms 2 and 4).
    """

    @staticmethod
    def forward(ctx, q, k, v,
                causal: bool = False,
                key_padding_mask: Tensor | None = None,
                dropout_p: float = 0.0,
                softmax_scale: float | None = None,
                block_q: int = 128,
                block_k: int = 128):
        # Save RNG state for dropout reproducibility
        cpu_state, cuda_state = _save_rng_state_for_device(q.device)

        out, m, l = flashattention_forward_paper(
            q, k, v,
            causal=causal,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            block_q=block_q,
            block_k=block_k,
        )

        # Save for backward
        # Note: saving q,k,v increases memory but keeps code clear and paper-aligned.
        ctx.save_for_backward(q, k, v, out, m, l, key_padding_mask if key_padding_mask is not None else torch.tensor([], device=q.device))
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.block_q = block_q
        ctx.block_k = block_k
        ctx.has_kpm = key_padding_mask is not None
        ctx.cpu_state = cpu_state
        ctx.cuda_state = cuda_state
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, m, l, kpm_saved = ctx.saved_tensors
        causal = ctx.causal
        dropout_p = ctx.dropout_p
        softmax_scale = ctx.softmax_scale
        block_q = ctx.block_q
        block_k = ctx.block_k
        key_padding_mask = kpm_saved if ctx.has_kpm else None

        # Restore RNG state so dropout masks match forward exactly
        _restore_rng_state_for_device(q.device, ctx.cpu_state, ctx.cuda_state)

        dQ, dK, dV = flashattention_backward_paper(
            q, k, v, out, m, l, dout,
            causal=causal,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            block_q=block_q,
            block_k=block_k,
        )
        return dQ, dK, dV, None, None, None, None, None, None


def flashattention_paper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    block_q: int = 128,
    block_k: int = 128,
) -> Tensor:
    """
    User-facing function.
    """
    return FlashAttentionPaperV1.apply(q, k, v, causal, key_padding_mask, dropout_p, softmax_scale, block_q, block_k)







