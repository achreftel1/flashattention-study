# flashattention-study

## Day 1: Baseline Attention (Transformer)
Implements scaled dot-product attention:
Attn(Q,K,V) = softmax(QK^T / sqrt(d)) V

- src/attn_baseline.py: baseline attention + simple MHA wrapper
- tests/: correctness vs PyTorch SDPA + gradcheck

Equation source: Attention Is All You Need (Vaswani et al., 2017).