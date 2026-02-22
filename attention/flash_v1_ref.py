# attention/flash_v1_ref.py
import math
import torch
 
def flashattn_v1_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    block_q: int = 128,
    block_kv: int = 128,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Reference FlashAttention v1 forward: exact attention via tiling + online softmax.
    q, k, v: [B, H, L, D]
    returns: [B, H, L, D]
 
    Key idea (paper): avoid materializing [L, L] attention matrix by processing blocks
    and maintaining online softmax stats per row.  (m, l) 
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H, L, D = q.shape
    assert k.shape == (B, H, L, D)
    assert v.shape == (B, H, L, D)
 
    scale = (1.0 / math.sqrt(D)) if softmax_scale is None else softmax_scale
 
    # We'll accumulate in fp32 for stability (common practice for softmax workloads)
    out = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
 
    # Process Q in blocks (block_q) to mimic SRAM tiling in the paper
    for q_start in range(0, L, block_q):
        q_end = min(q_start + block_q, L)
        q_blk = q[:, :, q_start:q_end, :]  # [B,H,Bq,D]
 
        # Online softmax stats per row in this Q-block:
        # m: running max, l: running sum of exp (normalizer)
        m = torch.full((B, H, q_end - q_start), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, q_end - q_start), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, q_end - q_start, D), device=q.device, dtype=torch.float32)
 
        # Now iterate over KV blocks
        for kv_start in range(0, L, block_kv):
            kv_end = min(kv_start + block_kv, L)
            k_blk = k[:, :, kv_start:kv_end, :]  # [B,H,Bk,D]
            v_blk = v[:, :, kv_start:kv_end, :]  # [B,H,Bk,D]
 
            # Compute scores for this tile: [B,H,Bq,Bk]
            # scores = (Q_block @ K_block^T) * scale
            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale
 
            # Apply causal mask if needed:
            # If causal, query position i cannot attend to key position j > i.
            # Here query indices are [q_start, q_end), key indices [kv_start, kv_end)
            if is_causal:
                qi = torch.arange(q_start, q_end, device=q.device)[:, None]        # [Bq,1]
                kj = torch.arange(kv_start, kv_end, device=q.device)[None, :]      # [1,Bk]
                mask = kj <= qi  # [Bq,Bk]
                scores = scores.masked_fill(~mask, float("-inf"))
 
            # Online softmax update:
            # 1) find max of this block per row
            m_blk = scores.max(dim=-1).values.to(torch.float32)  # [B,H,Bq]
 
            # 2) new running max
            m_new = torch.maximum(m, m_blk)
 
            # 3) rescale old sums/outputs
            alpha = torch.exp(m - m_new)  # [B,H,Bq]
 
            # 4) compute exp(scores - m_new)
            p = torch.exp(scores.to(torch.float32) - m_new[..., None])  # [B,H,Bq,Bk]
 
            # 5) update normalizer l
            l_new = l * alpha + p.sum(dim=-1)  # [B,H,Bq]
 
            # 6) update output accumulator
            # o_new = o_old * alpha + p @ V_block
            o = o * alpha[..., None] + torch.matmul(p, v_blk.to(torch.float32))
 
            # commit updates
            m = m_new
            l = l_new
 
        # final normalization for this Q-block
        o = o / l[..., None]
        out[:, :, q_start:q_end, :] = o
 
    return out.to(q.dtype)