+++
title = "From Encoder to Decoder: Extending FLARE to Memory-Efficient Causal Attention"
date = 2026-02-18T00:00:00-05:00
draft = false
description = "How FLARE evolves from bidirectional encoder attention to scalable causal decoder-only training and inference."
ShowToc = true
TocOpen = true
math = true
+++

## Motivation

FLARE was originally developed as an encoder-style global mixing primitive: learned latent queries gather information from many tokens, then scatter it back. The decoder setting is harder because causality changes algorithmic dependencies, numerical stability constraints, and what efficiency means in training versus inference.

This post summarizes a practical path to causal FLARE for long-context language modeling. See also the [dissertation proposal talk](https://www.youtube.com/watch?v=8h9EXJqQUi0) for broader context.

---

## What changes from encoder to decoder?

Encoder attention is bidirectional: token $t$ can depend on any token $\tau$. Decoder attention is causal: token $t$ may depend only on $\tau \le t$.

This implies three requirements:

1. No future-token leakage.
2. Efficient prefill for long contexts.
3. Fast incremental decode with cached state.

---

## Recap: encoder FLARE as gather-scatter

Let latent queries be $Q_L \in \mathbb{R}^{M \times D}$, and token keys/values be $K,V \in \mathbb{R}^{N \times D}$.

$$
Z = \mathrm{SDPA}(Q_L, K, V), \qquad
Y = \mathrm{SDPA}(K, Q_L, Z).
$$

Interpretation:
- Gather: latents pool from tokens.
- Scatter: tokens read from latents.

---

## Causal softmax attention baseline

$$
y_t
= \sum_{\tau \le t}
\frac{\exp(q_t^\top k_\tau)}{\sum_{u \le t}\exp(q_t^\top k_u)}v_\tau
$$

or

$$
Y = \mathrm{softmax}\!\big((QK^\top) \odot M_{\mathrm{causal}}\big)V,
$$

where $M_{\mathrm{causal}}$ masks the strict upper triangle.

---

## Warm-up: causal linear attention as a state update

$$
y_t = \frac{S_t q_t}{q_t^\top z_t},
\qquad
S_t = \sum_{\tau \le t} v_\tau k_\tau^\top,
\qquad
z_t = \sum_{\tau \le t} k_\tau.
$$

This shows the decoder-friendly pattern: maintain prefix state, update incrementally, and compute $y_t$ from state plus $q_t$.

---

## Causal FLARE definition

A causal latent update can be written as:

$$
z_m^t =
\sum_{\tau \le t}
\frac{\exp(q_m^\top k_\tau)}{\sum_{u \le t}\exp(q_m^\top k_u)}v_\tau.
$$

Then token output at step $t$:

$$
y_t = \sum_{m=1}^M
\frac{\exp(k_t^\top q_m)}{\sum_{m'=1}^M \exp(k_t^\top q_{m'})}
z_m^t.
$$

So each step produces an updated latent set $Z_t = [z_1^t,\ldots,z_M^t]$, then token $t$ reads from it.

---

## Algorithm 1: streaming recurrent causal FLARE (decode-friendly)

Each latent maintains running online-softmax statistics across the token stream: a running max $\mu_{t,m}$, a normalizing denominator $d_{t,m}$, and a numerator accumulator $U_{t,m,:}$. The update at each step is:

1. Initialize:
   - $U_0 \in \mathbb{R}^{M \times D} \leftarrow 0$
   - $d_0 \in \mathbb{R}^{M} \leftarrow 0$
   - $\mu_0 \in \mathbb{R}^{M} \leftarrow -\infty$
2. For each token $t=1,\ldots,T$, given $(k_t, v_t)$:
   - $s_t \leftarrow (Qk_t)s$
   - $\mu_t \leftarrow \max(\mu_{t-1}, s_t)$
   - $\gamma \leftarrow \exp(\mu_{t-1}-\mu_t)$
   - $\eta \leftarrow \exp(s_t-\mu_t)$
   - $d_t \leftarrow d_{t-1}\odot\gamma + \eta$
   - $U_t \leftarrow U_{t-1}\odot\gamma + \eta\,v_t^\top$
   - $Z_t \leftarrow U_t \oslash d_t$
   - $\alpha_t \leftarrow \mathrm{softmax}(s_t)$
   - $y_t \leftarrow \alpha_t^\top Z_t$
3. Return $\{y_t\}_{t=1}^T$.

This recurrent form is ideal for autoregressive decode — cached state is updated in $O(M)$ work per token. However, it is throughput-inefficient for training because the backward pass must store all prefix statistics naively. The next two algorithms address the prefill and training setting.

---

## Algorithm 2: dense causal FLARE (prefill-oriented)

Define scores:

$$
S_{t,m} = s\,k_t^\top q_m,
\qquad
A_{t,m} = \exp(S_{t,m}),
\qquad
P_{t,m} = \mathrm{softmax}_m(S_{t,:}).
$$

Prefix denominator per latent:

$$
D_{t,m} = \sum_{u \le t} A_{u,m}.
$$

Then

$$
C_{t,m} = \frac{P_{t,m}}{D_{t,m}+\varepsilon},
\qquad
W = C A^\top,
\qquad
Y = (W \odot M_{\mathrm{causal}})V.
$$

This dense form is matmul-friendly and efficient for training, but computing $\exp(S)$ directly is numerically unstable for long contexts in mixed precision — exponents of large scores overflow in BF16. Algorithm 3 fixes this.

---

## Algorithm 3: stable dense causal FLARE

Use online-softmax style prefix statistics for each latent:

- $R_{t,m}$: running prefix max of $S_{t,m}$
- $L_{t,m}$: stable prefix sum in normalized frame

$$
R_{t,m} = \max(R_{t-1,m}, S_{t,m}),
$$

$$
L_{t,m} = L_{t-1,m}\exp(R_{t-1,m}-R_{t,m}) + \exp(S_{t,m}-R_{t,m}).
$$

Then

$$
C_{t,m} = \frac{P_{t,m}}{L_{t,m}+\varepsilon},
\qquad
W_{t,\tau} = \sum_{m=1}^M C_{t,m}\exp(S_{\tau,m}-R_{t,m}),
$$

$$
W \leftarrow W \odot M_{\mathrm{causal}},
\qquad
Y = WV.
$$

This keeps the prefill path numerically stable while preserving dense-kernel structure.

1. Compute score and latent decode probabilities:
   - $S \leftarrow s(KQ^\top)$, so $S_{t,m}=s\,k_t^\top q_m$
   - $P \leftarrow \mathrm{softmax}_m(S)$
2. Compute stable prefix statistics for each latent:
   - Initialize $R_{0,m}\leftarrow -\infty,\;L_{0,m}\leftarrow 0$
   - For $t=1,\ldots,T$:
     - $R_{t,m}\leftarrow \max(R_{t-1,m}, S_{t,m})$
     - $L_{t,m}\leftarrow L_{t-1,m}\exp(R_{t-1,m}-R_{t,m})+\exp(S_{t,m}-R_{t,m})$
3. Build dense causal mixer:
   - $C_{t,m}\leftarrow \dfrac{P_{t,m}}{L_{t,m}+\varepsilon}$
   - $W_{t,\tau}\leftarrow \sum_{m=1}^M C_{t,m}\exp(S_{\tau,m}-R_{t,m})$
   - $W \leftarrow W \odot M_{\mathrm{causal}}$
4. Output:
   - $Y \leftarrow WV$.

---

## Train vs prefill vs decode

Causal FLARE supports three operational regimes, each with different priorities.

**Teacher-forced training** processes the full sequence in parallel and is throughput-oriented. Algorithm 3 is the right choice: stable prefix statistics, chunking over time to avoid materializing $T \times T$, and fused kernels for arithmetic intensity.

**Inference prefill** is algorithmically identical to training but with a different blocking profile. Prefill is latency-sensitive and may benefit from different tile sizes and more aggressive kernel fusion than the training path.

**Autoregressive decode** is latency-critical and processes one token at a time. Algorithm 1 is ideal: update the cached latent state with each new $(k_t, v_t)$, then read $y_t$ from the updated latents. No attention matrix is ever constructed, and the state size $M$ is the only memory overhead beyond the KV cache.

---

## Adaptive attention state size

A practical FLARE advantage is controllable latent/state budget:

- Larger state in prefill for fidelity
- Smaller state in decode for throughput

This exposes a direct compute-memory-accuracy knob.

---

## Systems notes

A few implementation details matter disproportionately.

**FP32 prefix accumulators.** The running max and sum statistics accumulate error across many tokens. Accumulating in FP32 prevents catastrophic cancellation even when inputs are in BF16 or FP16.

**Time chunking.** Processing time in chunks avoids materializing the full $T \times T$ intermediate — which is precisely the goal. Chunk size trades register pressure against memory traffic and should be tuned per GPU.

**Separate kernels per regime.** Training, prefill, and decode have different access patterns and arithmetic intensities. A single fused kernel cannot be optimally tiled for all three; separate kernels let you autotune each independently.

**Memory bandwidth first.** At long contexts, causal FLARE is memory-bandwidth-bound rather than compute-bound. Optimizing cache layout and minimizing global memory traffic matters more than maximizing FLOPs/s.

---

## References

1. Puri, V. et al. *FLARE: Fast Low-rank Attention Routing Engine*. arXiv (2025). [https://arxiv.org/abs/2508.12594](https://arxiv.org/abs/2508.12594)
2. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS (2017). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS (2022). [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
4. Qin, Z. et al. *The Devil in Linear Transformer*. arXiv (2022). [https://arxiv.org/abs/2210.10340](https://arxiv.org/abs/2210.10340)
