# flashattention-study — Attention → FlashAttention (v1) from scratch (PyTorch)

This repository is a **study + from-scratch implementation** of:

1. **Scaled Dot-Product Attention** (Transformer baseline)
2. **FlashAttention v1**: *exact* attention computed with **tiling + online softmax** (forward + backward)
3. **Why FlashAttention-2 exists**: limitations of v1 and the FA2 improvements (parallelism/work partitioning)

The goal is to be able to explain **every equation** and how it maps to code, then benchmark and compare memory/time behavior.

---

## Quick summary

* Implements a readable PyTorch reproduction of the algorithmic math behind (1) baseline scaled dot-product attention and (2) FlashAttention v1 (exact, IO-aware, tiled, online-softmax).
* Includes forward + backward (recompute) implementations, correctness tests, and benchmarks (runtime + peak memory).

---

## Repo layout

```
.
├── attention/          # all source code (baseline + flash v1)
├── bench/              # benchmarking scripts/results (runtime + memory)
└── tests/              # correctness tests (forward + backward)
```

Recommended files inside `attention/`:

* `attention/attn_baseline.py` — baseline scaled dot-product attention
* `attention/flash_v1.py` — FlashAttention v1 (forward + custom backward)
* `attention/utils.py` — helper routines (masking, tiling utils, numerics)

---

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run tests (auto CPU/GPU):

```bash
pytest -q
```

3. (Windows PowerShell) Force GPU on Windows:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; pytest -q
```

> Note: these instructions assume CUDA is available when you want GPU tests.

---

# Part A — Transformer Attention (Baseline)

## A1) Scaled dot-product attention (equation)

The core formula used throughout:

[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right) V
]

Shapes used in this repo:

* (Q \in \mathbb{R}^{B\times H\times L\times D}) — queries
* (K, V \in \mathbb{R}^{B\times H\times S\times D}) — keys and values

Where: B = batch, H = heads, L = query length, S = key/value length, D = head dimension.

## A2) Masks

* **Causal mask**: autoregressive attention (token i can attend only to keys j ≤ i).
* **Padding mask**: prevent attention to padding positions in the key/value sequence.

The baseline implementation explicitly builds the score matrix and then applies softmax + matmul with V. This is straightforward but requires O(L×S) memory for the score/probability tensors per head.

---

# Part B — FlashAttention v1 (Exact + IO-aware)

FlashAttention v1 is an *exact* attention algorithm designed to reduce device memory traffic by computing attention in tiles and using an online, numerically-stable softmax that never materializes the full L×S attention matrix.

## B1) Why standard attention is inefficient

Standard attention often materializes the full score matrix (S=QK^\top/\sqrt{D}) or the probability matrix (P=\mathrm{softmax}(S)), each sized L×S. For long sequences this becomes the dominant memory and IO cost. FlashAttention avoids storing those large intermediates by processing keys/values in blocks.

## B2) FlashAttention forward (tiling + online softmax)

We want to compute:

[
S = \frac{Q K^\top}{\sqrt{D}},\quad P = \mathrm{softmax}(S),\quad O = P V.
]

Instead of forming full (S) or (P), we iterate over key/value blocks and maintain per-query running statistics to perform an *online softmax* per row. For each query row `i` we maintain:

* running max (m_i)
* running normalizer (sum of exponentials) (\ell_i)
* running numerator accumulator (o_i) (accumulates weighted value sums)

When processing a new key block we:

1. compute block scores (s_{i,blk}) (this block's dot products for all queries)
2. compute new running max (m_i^{new} = \max(m_i, \max s_{i,blk}))
3. rescale previous (\ell_i, o_i) by (e^{m_i - m_i^{new}})
4. compute block exponentials (p_{i,blk} = e^{s_{i,blk} - m_i^{new}})
5. update (\ell_i \leftarrow \ell_i + \sum p_{i,blk}) and (o_i \leftarrow o_i + p_{i,blk} V_{blk})

After processing all blocks, the output per query is (O_i = o_i / \ell_i).

This yields numerically-stable, exact softmax results computed blockwise while avoiding L×S intermediates.

## B3) What we store for backward

To enable backward without storing the full probability matrix, FlashAttention saves small per-query statistics (commonly the `log-sum-exp` per row, (\mathrm{lse}*i = \log \sum_j e^{s*{ij}})). When recomputing backward, we re-create block probabilities with

[P_{ij} = \exp(s_{ij} - \mathrm{lse}_i)]

and compute gradients blockwise, accumulating gradients for Q, K, and V.

---

# Part C — FlashAttention v1 Backward (recompute blocks)

Given (O = P V) with (P = \mathrm{softmax}(S)) and (S = QK^\top / \sqrt{D}), the backward pass computes gradients for Q, K, V without materializing (P):

* (dV = P^\top dO)
* (dP = dO , V^\top)

Use the softmax Jacobian row-wise to relate (dP) to (dS):

[dS = P \odot (dP - \delta),\quad \delta_i = \sum_j P_{ij} dP_{ij}]

A common simplification used in the derivation is

[\delta_i = dO_i \cdot O_i,]

which follows from algebraic rearrangement when substituting `dP = dO V^T` and using properties of row-wise weighted sums. The backward algorithm recomputes block scores, reconstructs (P_{blk}) from stored `lse`, and accumulates `dQ`, `dK`, and `dV` blockwise.

---

# Part D — FlashAttention-2: Why it exists

FlashAttention v1 already gives major IO/memory improvements, but there is additional GPU performance headroom. Limitations of v1 include work partitioning and shared-memory synchronization that can leave GPU compute units underutilized. FlashAttention-2 focuses on engineering improvements to:

* reduce non-GEMM FLOPs and overhead
* parallelize attention across thread blocks (improve occupancy)
* distribute work across warps to reduce shared-memory communication

The highly-optimized CUDA kernels for these variants live in the official repo; study of FA2 focuses on mapping algorithmic ideas to kernel engineering.

---

# Part E — Equation → Code mapping (defendable points)

This section lists the most important equations and how they map to the code in `attention/`.

## E1) Baseline attention

Equation:

[O = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right) V]

Code mapping (in `attention/attn_baseline.py`):

* `scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)`
* `prob = torch.softmax(scores, dim=-1)`
* `out = torch.matmul(prob, V)`

## E2) FlashAttention v1 forward

Concept → code mapping (in `attention/flash_v1.py` forward):

* tile keys/values by `blk`
* compute block scores `s_blk = Q @ K_blk^T / sqrt(D)`
* update running `m`, `l`, `o` per query using the online-softmax rules (rescale old accumulators, add new exponentials and value-weighted sums)
* final `out = o / l`

## E3) FlashAttention v1 backward

Concept → code mapping (in `attention/flash_v1.py` backward):

* recompute block scores from Q and K
* reconstruct `P_blk = exp(s_blk - lse)` using stored `lse` (or recomputed per-row lse)
* compute `dV_blk = P_blk^T @ dO_block` and accumulate
* compute `dP_blk = dO_block @ V_blk^T` and then `dS_blk = P_blk * (dP_blk - delta)` with appropriate `delta` per row
* accumulate `dQ` and `dK` via matmuls with `dS_blk`

---

# Part F — Validation

**Forward**: outputs must match baseline attention within numerical tolerance.
**Backward**: gradients must match baseline autograd; run `torch.autograd.gradcheck` on CPU float64 for extra confidence.

FlashAttention v1 is algorithmically exact (not approximation); the tests verify equivalence.

---

# Part G — Benchmarking (time + peak memory)

Benchmarks live in `bench/` and should include:

* forward runtime vs sequence length
* backward runtime vs sequence length
* peak GPU memory (use `torch.cuda.max_memory_allocated()`)

Typical plots: runtime and peak memory vs sequence length for baseline vs FlashAttention v1. Use consistent hardware and `torch.cuda.reset_peak_memory_stats()` between runs.

---

# Part H — Study path (10-step plan)

1. Implement baseline attention and confirm shapes/masks.
2. Explain why attention is a bottleneck for long sequences (quadratic intermediates).
3. Learn GPU memory-hierarchy intuition (HBM vs SRAM) and IO-awareness.
4. Implement FlashAttention forward: tiling + online softmax.
5. Add causal + padding mask correctness to FlashAttention.
6. Implement FlashAttention backward: recompute using stored `lse`.
7. Validate gradients vs baseline and run gradcheck.
8. Benchmark runtime and peak memory; plot results.
9. Study why FA1 is not fully GEMM-efficient (occupancy/work partitioning).
10. Read FA2 engineering notes and map them to the official CUDA kernels.

---

# Part I — Relation to the official implementation

The highly-optimized CUDA kernels and production-level code live in the official implementation of the algorithm (see References). This study repo focuses on readable PyTorch reproductions of the algorithmic math and correctness proofs.

---

# Two small improvements to add (supervisor-ready)

* **Add a “Presentation Defense” section in the README**: terse checklist explaining the online softmax updates, why `lse` suffices for backward, the delta trick ((\delta_i = dO_i \cdot O_i)), and why FA2 exists (occupancy/partitioning).
* **Ensure code is pushed to the public repo**: make sure the `attention/`, `bench/`, `tests/` folders are present on the remote and document the commit steps in the README.

Suggested push commands (run locally):

```bash
git add attention bench tests
git commit -m "Add baseline attention + FlashAttention v1 forward/backward + tests + bench"
git push
```

---

# How to cite

If you use results or ideas, cite the original papers (search the preprint server for the canonical arXiv entries for each paper):

* FlashAttention v1 (paper)
* FlashAttention-2 (paper)
* Transformer ("Attention Is All You Need")

---

# References

* **entity["organization","Dao-AILab/flash-attention","github repo"]** — official CUDA implementations and releases
* **entity["organization","Hugging Face","nlp org"]** — community writeups and README-like descriptions
* **entity["organization","PyTorch","deep learning framework"]** — blog posts and framework notes about FlashAttention-2
* **entity["organization","GitHub","code hosting"]** — general repo hosting
* **entity["organization","Gitee","chinese code hosting"]** — mirror of the official implementation
* **entity["organization","arXiv","preprint server"]** — canonical preprints

---

## License

Include your preferred license file in the repo (`LICENSE` with MIT).

---
