+++
title = "Triple Attention in Triton: Building a Third-Order Memory in Linear Time"
date = 2026-02-18T00:00:00-05:00
draft = false
description = "Designing and implementing third-order attention with Triton kernels and linear-time sequence scaling."
ShowToc = true
TocOpen = true
math = true
+++

## Motivation

Pairwise attention is powerful, but it compresses interaction structure into second-order forms. Most efficient attention methods try to approximate or factor the $N \times N$ attention matrix. Triple attention takes a different perspective:

> Instead of modeling pairwise token interactions, build a structured higher-order memory and let tokens read from it.

This post explains how triple attention works conceptually, and how we implement it in Triton using fused kernels that scale linearly in sequence length.

---

## Why third-order memory?

Standard linear attention builds a pooled memory $S \in \mathbb{R}^{D \times D}$:

$$ S_{ij} = \sum_{n=1}^{N} K_{ni} V_{nj}, $$

and predicts with:

$$ Y_{nj} = \sum_{i=1}^{D} Q_{ni} S_{ij}. $$

Triple attention instead accumulates a third-order state $S \in \mathbb{R}^{D_q \times D_v \times D_q}$:

$$ S_{ijk} = \sum_{n=1}^{N} K^{(1)}_{ni} V_{nj} K^{(2)}_{nk}, $$

with output:

$$ Y_{nj} = \sum_{i=1}^{D} \sum_{k=1}^{D} Q^{(1)}_{ni} S_{ijk} Q^{(2)}_{nk}. $$

The sequence reduction stays linear in $N$, while representational capacity expands from $D \times D$ to $D \times D \times D$. The tradeoff: compute scales as $O(N D_q^2 D_v)$, so this mechanism suits settings where $D_q$ is fixed and $N$ is large.

---

## From quadratic attention to structured memory

Standard self-attention forms $A = \text{softmax}(QK^\top)V$, which requires materializing or implicitly computing an $N \times N$ interaction. Triple attention avoids this entirely. Instead of computing token-to-token interactions, we construct a **third-order state tensor** $S \in \mathbb{R}^{D_q \times D_v \times D_q}$ that aggregates global information from all tokens in a streaming fashion. Each token then reads from this state via two learned query projections. No $N \times N$ matrix is ever formed.

---

## Tensor Shapes

We begin with projected tensors:

- $Q_1, Q_2, K_1, K_2 \in \mathbb{R}^{B \times H \times N \times D_q}$
- $V \in \mathbb{R}^{B \times H \times N \times D_v}$

For kernel simplicity, batch and heads are flattened:

```
Q1, Q2, K1, K2: [BH, N, Dq]
V:              [BH, N, Dv]
```

We allocate:

- `STATE: [BH, Dq, Dv, Dq]` (accumulated in fp32)
- `O:     [BH, N, Dv]`

The memory cost is independent of sequence length $N$.

---

## Forward pass

The forward pass is split into two phases, each corresponding to a fused Triton kernel.

### Phase 1: Streaming state construction

Each kernel instance tiles the indices $(i, j, k)$ of the state tensor, streams over tokens in chunks (e.g. 4096 at a time), and accumulates contributions into `STATE` using atomic adds in fp32.

$$
S_{ijk} = \sum_{n=1}^N K_{1,n,i} \cdot V_{n,j} \cdot K_{2,n,k}
$$

No token-token matrix is constructed; streaming over the sequence makes this linear in $N$. As a reference:

```python {linenos=false}
import torch

def build_triple_state(k1, k2, v):
    """
    Phase 1: accumulate the third-order global memory.

    Args:
        k1: [B, H, N, Dq]    first key projection
        k2: [B, H, N, Dq]    second key projection
        v:  [B, H, N, Dv]    value projection

    Returns:
        state: [B, H, Dq, Dv, Dq]    third-order state (fp32)
    """
    # S[b,h,i,j,k] = sum_n  k1[n,i] * v[n,j] * k2[n,k]
    return torch.einsum('bhni,bhnj,bhnk->bhijk',
                        k1.float(), v.float(), k2.float())
```

Complexity: **O(N D_q^2 D_v)** — linear in sequence length. The Triton kernel tiles over $(i,j,k)$ and streams over $n$ to avoid materializing the full intermediate.

---

### Phase 2: Output contraction

Each token reads from the precomputed `STATE` via two learned query projections:

$$
O_n = \sum_{i,j,k} Q_{1,n,i} \cdot S_{ijk} \cdot Q_{2,n,k}
$$

The state acts as a global communication hub — each token independently decides how to read from it via $Q_1$ and $Q_2$.

```python {linenos=false}
def read_triple_state(q1, q2, state):
    """
    Phase 2: contract each token's queries against the global state.

    Args:
        q1:    [B, H, N, Dq]         first query projection
        q2:    [B, H, N, Dq]         second query projection
        state: [B, H, Dq, Dv, Dq]   precomputed third-order memory

    Returns:
        y: [B, H, N, Dv]    token outputs
    """
    # y[b,h,n,j] = sum_{i,k}  q1[n,i] * state[i,j,k] * q2[n,k]
    return torch.einsum('bhni,bhijk,bhnk->bhnj',
                        q1.float(), state, q2.float())
```

This is structurally similar to routing through a learned latent space, but realized as a multilinear contraction.

### Full reference

```python {linenos=false}
import torch

def triple_attn(q1, q2, k1, k2, v):
    """
    Triple attention: linear-time third-order global memory.

    Args:
        q1: [B, H, N, Dq]    first query projection
        q2: [B, H, N, Dq]    second query projection
        k1: [B, H, N, Dq]    first key projection
        k2: [B, H, N, Dq]    second key projection
        v:  [B, H, N, Dv]    value projection

    Returns:
        y:  [B, H, N, Dv]    token outputs
    """
    state = build_triple_state(k1, k2, v)     # [B, H, Dq, Dv, Dq]
    y     = read_triple_state(q1, q2, state)  # [B, H, N, Dv]
    return y.to(v.dtype)
```

---

## Why this scales

Standard attention scales as:

$$
O(N^2 D)
$$

Triple attention scales as:

$$
O(N D_q^2 D_v)
$$

If $D_q$ is held fixed as $N$ grows, this is linear in sequence length.

Memory never grows with $N^2$.

This makes it viable for long-sequence workloads, including:

- PDE surrogate models
- Large point-cloud processing
- Long-context sequence modeling

---

## Numerical considerations

Several implementation details are critical:

- Accumulate `STATE` in **fp32**.
- Use tensor cores (TF32 where available).
- Store outputs in fp16/bf16.
- Chunk over sequence dimension to fit memory.
- Use atomic adds carefully to avoid race conditions.

The mixed-precision strategy preserves stability while keeping memory bandwidth low.

---

## Backward pass

The backward pass mirrors the forward decomposition.

Implemented as a custom `torch.autograd.Function`, it performs:

### 1. Accumulate dSTATE

Compute:

$$
dS = \sum_n dO_n \cdot Q_{1,n} \cdot Q_{2,n}
$$

Streaming over tokens in chunks.

### 2. Gradients for K1, K2, V

Contract `dSTATE` with remaining factors:

- `dK1`
- `dK2`
- `dV`

Each has its own fused Triton kernel.

### 3. Gradients for Q1, Q2

Contract saved `STATE` with `dO`.

The code includes explicit einsum expressions for gradient verification, making parity testing against a reference implementation straightforward.

---

## Conceptual perspective

Triple attention reflects a broader idea:

> Global communication does not require pairwise token interaction.

Instead of asking “which tokens attend to which?”, we ask:

> Can we compress global structure into a structured tensor, and let tokens read from it?

This viewpoint connects to:

- Low-rank attention
- Latent routing methods
- Multilinear tensor contractions
- Structured operator learning

It also suggests a hypothesis:

> If a structured self-attention mechanism captures global communication efficiently, it may extend naturally to causal and autoregressive settings.

Exploring that extension is ongoing work.

---

## Open problems

- Better approximations to reduce $D^3$ pressure
- Hybrid blocks combining low-rank gather–scatter with triple memory
- Causal decoding variants for language modeling

---

## Closing thoughts

Triple attention is not just a kernel experiment — it is an exploration of structured global memory.

By fusing state construction and output contraction in Triton, we obtain linear scaling in sequence length, stable mixed-precision execution, and a flexible multilinear attention primitive. This kernel serves as a foundation for further experiments in structured and adaptive attention mechanisms.

The full implementation is available in the FLARE repository alongside the paper ([arXiv:2508.12594](https://arxiv.org/abs/2508.12594)).

---

## References

1. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS (2017). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Katharopoulos, A. et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML (2020). [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
3. Kozachinskiy, A. et al. *Strassen Attention, Split VC Dimension and Compositionality in Transformers*. arXiv (2025). [https://arxiv.org/abs/2501.19215](https://arxiv.org/abs/2501.19215)
4. Roy, A. et al. *Fast and Simplex: 2-Simplicial Attention in Triton*. arXiv (2025). [https://arxiv.org/abs/2507.02754](https://arxiv.org/abs/2507.02754)
5. Qin, Z. et al. *The Devil in Linear Transformer*. arXiv (2022). [https://arxiv.org/abs/2210.10340](https://arxiv.org/abs/2210.10340)
