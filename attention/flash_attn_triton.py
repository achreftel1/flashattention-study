import math

import torch

import triton
import triton.language as tl

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)

class TritonAttention1(torch.autograd.Function):
    """
    Phase 1:
      forward  = Triton kernel (blockwise + online softmax)
      backward = PyTorch recomputation backward (your proven-correct version)

    This mirrors Dao-AILab's overall Triton structure (kernels + wrapper + autograd),
    but starts with a stable backward so you can progress fast. [1](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/sharma_hocevar_flashattention.pdf)[2](https://huggingface.co/papers/2205.14135)
    """

    @staticmethod
    def forward(ctx, Q, K, V, causal: bool = False, softmax_scale: float | None = None):
        """
        Q: (batch_size, seqlen_q, nheads, headdim)
        K, V: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(Q.shape[-1])

        out, lse = _flash_attn_triton_forward(Q, K, V, causal=causal, softmax_scale=softmax_scale)

        # Save for backward
        ctx.save_for_backward(Q, K, V, out, lse)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        Q, K, V, out, lse = ctx.saved_tensors
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale

        # IMPORTANT:
        # For Phase 1 we reuse known-correct recomputation backward.
        # Implement this function in attention/flash_v1.py  (already have it conceptually),
        # or create a shared helper: attention/flash_backward_reference.py
        from .flash_v1 import _reference_flash_backward  #  create this helper

        dQ, dK, dV = _reference_flash_backward(
            Q, K, V, out, lse, dout,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        return dQ, dK, dV, None, None


def triton_attention1(Q, K, V, causal: bool = False, softmax_scale: float | None = None):
    """
    User-facing API matching Dao-style flash_attn_func(q,k,v,...),
    but only for separate q,k,v layout. [1](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/sharma_hocevar_flashattention.pdf)[2](https://huggingface.co/papers/2205.14135)
    """
    return TritonAttention1.apply(Q, K, V, causal, softmax_scale)

