+++
title = "Higher-Order Attention in Linear Time: Multilinear Memories and Simplex Mixing"
date = 2026-02-16T00:00:00-05:00
draft = false
description = "A practical tour of linear attention, its bottlenecks, and why multilinear / simplex-style attention might help"
ShowToc = true
TocOpen = true
math = true
+++

## Beyond pairwise attention

Softmax attention is an extremely expressive token-mixing primitive, but it is expensive. For a sequence of length $N$, the core interaction matrix $Q \cdot K^\top$ is $N \times N$, which drives both runtime and memory. Linear transformers try to keep global context while avoiding the quadratic scaling. The catch is that the simplest linear attention formulations often lose a key ingredient that makes softmax attention work so well: **token-specific routing**.

This post has two goals. First, I want to explain why vanilla linear attention underperforms in many settings, not as a matter of “bad approximation,” but as a structural consequence of how the computation is arranged. Second, I want to sketch a set of experimental ideas I’ve been exploring around **enhanced attention mechanisms**. To be explicit: these are research notes. Some of these ideas are promising on paper, but they have *not* reliably panned out in my experiments so far. I’m writing them up anyway because the framing has been useful for me, and it may help others reason about the design space.

One additional constraint threads through the post: **permutation equivariance**. Many strong long-sequence models exploit 1D sequence structure (chunking, convolution, hierarchical pooling, scan recurrences). That is great for language, but it is brittle for unstructured grids and point sets where there is no canonical ordering and where I would like the model to be insensitive to permutations.

---

## Preliminaries

### Vanilla softmax attention (SDPA)

Let $X \in \mathbb{R}^{N \times C}$ be a sequence of $N$ tokens with $C$ features. We form queries/keys/values by linear projection:

$$Q = XW^q,\quad K = XW^k,\quad V = XW^v$$

where $W^q,W^k,W^v \in \mathbb{R}^{C \times C}$. In multi-head attention we split features into $H$ heads of dimension $d = C/H$, writing

$$Q=[Q_1,\dots,Q_H],\quad K=[K_1,\dots,K_H],\quad V=[V_1,\dots,V_H].$$

Scaled dot-product attention (SDPA) in head $h$ is

$$Y_h = \mathrm{SDPA}(Q_h,K_h,V_h;s) = \mathrm{softmax}\!\left(\frac{Q_hK_h^\top}{s}\right)V_h$$

with $s \approx \sqrt{d}$.
In vector form,

$$y_i = \sum_{j=1}^N \frac{\exp(q_i^\top k_j)}{\sum_{\ell=1}^N \exp(q_i^\top k_\ell)}\cdot v_j.$$

The quadratic cost comes from the pairwise interaction matrix $QK^\top \in \mathbb{R}^{N\times N}$.
A useful mental model is: each query $q_i$ produces its own distribution over keys, so it can route information in a token-specific way. That “personalized routing” is what many linear-time mechanisms struggle to preserve.

```python {linenos=false}
import torch

def sdpa_naive(q, k, v):
    """
    Scaled dot-product attention.

    Args:
        q: [B, H, N, D]    queries (already split across heads)
        k: [B, H, N, D]    keys
        v: [B, H, N, D]    values
        scale: float       attention scale (default 1.0)

    Returns:
        y: [B, H, N, D]    output
    """
    scale = q.size(-1) ** -0.5

    S = (q @ k.mT) * scale # [B, H, N, N] attn logits
    A = S.softmax(dim=-1)  # [B, H, N, N] attn weights
    y = A @ v              # [B, H, N, D]
    
    return y
```

---

## Linear attention

### The basic idea

Linear attention replaces the softmax kernel with a feature map $\phi(\cdot)$ (often constrained to be positive) so that similarities become dot products in feature space.
Define

$$\tilde{Q}=\phi(Q),\quad \tilde{K}=\phi(K).$$

A typical (row-normalized) update is

$$Y = \mathrm{rownorm}(\tilde Q\tilde K^\top)V,$$

which in vector form is

$$y_i = \sum_{j=1}^N \frac{(\tilde q_i^\top \tilde k_j)}{\sum_{\ell=1}^N \tilde q_i^\top \tilde k_\ell}\cdot v_j.$$

The efficiency comes from associativity:

$$y_i = \frac{\tilde q_i^\top\left(\sum_{j=1}^N \tilde k_j v_j^\top\right)}{\tilde q_i^\top\left(\sum_{\ell=1}^N \tilde k_\ell\right)}.$$

If we define a pooled “memory” state $S=\tilde K^\top V \in \mathbb{R}^{D\times D}$ and $z=\tilde K^\top\mathbf{1}\in\mathbb{R}^D$, we can write $Y = \dfrac{\tilde Q \cdot S}{\tilde Q \cdot z}$. This is linear in $N$ (up to constants) because the expensive terms are reductions over the sequence.
At a systems level, linear attention is appealing: you can stream over tokens, maintain a running state, and avoid materializing $N\times N$ matrices.

```python {linenos=false}
import torch

def linear_attn(q, k, v, kernel=None):
    """
    Linear attention: linear-time global mixing via associative state reduction.

    Args:
        q:      [B, H, N, D]    queries
        k:      [B, H, N, D]    keys
        v:      [B, H, N, D]    values
        kernel: callable        feature map φ applied to q and k (e.g. elu + 1)

    Returns:
        y: [B, H, N, D]    output
    """
    if kernel is not None:
        q = kernel(q)
        k = kernel(k)

    state = k.mT @ v                    # [B, H, D, D]
    k_sum = k.sum(dim=2).unsqueeze(-1)  # [B, H, D, 1]

    num = q @ state
    den = q @ k_sum
    return num / (den + 1e-6)
```

---

## Why vanilla linear attention often underperforms

It helps to look at the structural form of the update and ignore normalization for intuition. Linear attention can be written as 

$$y_i = \tilde q_i^\top\left(\sum_{j=1}^N \tilde k_j v_j^\top\right),$$

i.e., $y_i = \tilde q_i^\top S$ where $S\in\mathbb{R}^{D\times D}$.

The critical observation is simple: $\tilde q_i$ changes with $i$, but the memory $S$ is shared across all tokens. Every token is querying the same pooled summary of key/value content. That makes the model efficient, but it also changes the nature of token mixing.

**Token-specific routing is weakened.** In softmax attention, each query produces its own distribution over keys. In basic linear attention, each query projects against the same global state. You can still get token-dependent outputs because $\tilde q_i$ varies, but the mechanism no longer has “choose a few values and ignore the rest” behavior in the same way.

**Smoothing is hard to avoid.** The associativity trick forces compression before interaction. Fine-grained information has to survive the bottleneck $S$ to influence outputs. This tends to smear distinctions unless the feature map and projections work very hard to preserve them.

**The bottleneck lives in feature space.** Softmax attention carries $O(N^2)$ interaction capacity through an $N\times N$ attention map. Linear attention collapses this into a feature-space quadratic form: $S \in \mathbb{R}^{D\times D}$. Unless $\phi$ is carefully chosen (e.g., kernelized approximations of $\exp$), this can be structurally less expressive, not merely a “worse approximation.”

This is why the linear-transformer literature has so many “patches”: better feature maps (Performer/FAVOR+ style), gating and recurrence (RetNet/RWKV-like ideas), low-rank bottlenecks (Nyström, latent tokens), and various normalization tricks (including the practical observation that removing row-normalization and stabilizing with RMSNorm at the end can sometimes help).

---

## Higher-order attention: why pairwise may not be enough

Softmax attention is pairwise: it scores $(q_i,k_j)$ and uses that to weight $v_j$. Many tasks can be solved with pairwise interactions plus depth, but there are reasons to explore **explicit multi-way mixing**.

On algorithmic tasks, it is natural to point at interactions that look like (digit1, digit2, carry). On PDE-like data, nonlinearities often couple multiple factors, and it is tempting to ask whether explicitly modeling multi-way interactions could reduce the burden on depth or help linear-time models recover some selectivity.

One concrete formalization is **2-simplex (3-way) attention**, which replaces bilinear dot products with trilinear forms. I do not view this as “the answer,” but it is a clean starting point for thinking about higher-order token mixing.

---

## 2-simplex attention (3-way interactions)

Standard attention is built on the bilinear form

$$\langle x_1,x_2\rangle = \sum_{d=1}^D x_1[d]x_2[d].$$

2-simplex attention generalizes this to a trilinear form

$$\langle x_1,x_2,x_3\rangle = \sum_{d=1}^D x_1[d]x_2[d]x_3[d].$$

A naive 3-way attention mechanism defines a score tensor over triples. Let

$$Q,K^1,K^2,V^1,V^2 \in \mathbb{R}^{N\times D},$$

and define

$$S_{ijk}=\sum_{d'=1}^D Q_{id'}K^1_{jd'}K^2_{kd'}.$$

Then $A=\mathrm{softmax}(S)$ (softmax over $(j,k)$) and

$$Y_{id}=\sum_{j,k=1}^N A_{ijk}V^1_{jd}V^2_{kd}.$$

This is expressive, but it requires forming $N\times N\times N$ tensors, which is intractable for large $N$.
So the question that keeps coming up is: can we capture some higher-order behavior without paying cubic cost?

---

## Proposal: multilinear kernelized attention (linear-time simplex-style mixing)

This is where the “research notes” part begins. The idea below is not a claim that I have a working model. It is a proposal that falls out of the same associativity trick that makes linear attention fast. It is conceptually clean, but in my experiments it has been finicky to stabilize and has not consistently beaten simpler baselines.

The key trick in linear attention is: avoid instantiating $N\times N$ by collapsing into a $D\times D$ state. We can do something similar for 2-simplex attention by dropping softmax and restructuring the computation so the triple interaction factorizes into feature-space memories.

Start from the unnormalized 3-way update:

$$A_{ijk}=\sum_{d'=1}^D Q_{id'}K^1_{jd'}K^2_{kd'},$$

and

$$Y_{id}=\sum_{j,k=1}^N A_{ijk}V^1_{jd}V^2_{kd}.$$

Rearranging sums gives

$$Y_{id}=\sum_{d'=1}^D Q_{id'}\left(\sum_{j=1}^N K^1_{jd'}V^1_{jd}\right)\left(\sum_{k=1}^N K^2_{kd'}V^2_{kd}\right).$$

Define feature-space memories

$$S^1=(K^1)^\top V^1 \in \mathbb{R}^{D\times D}$$

and

$$S^2=(K^2)^\top V^2 \in \mathbb{R}^{D\times D}.$$

Then the output becomes $Y = Q\left(S^1 \odot S^2\right)$ where $\odot$ is the elementwise Hadamard product. This is linear in $N$ because forming each $S^\ell$ is a single reduction pass over tokens.

### Generalization to L-way multilinear attention

With $L$ key/value pairs

$$\{(K^\ell,V^\ell)\}_{\ell=1}^L,$$

define

$$S^\ell=(K^\ell)^\top V^\ell \in \mathbb{R}^{D\times D}$$

and compute

$$Y = Q\left(S^1 \odot S^2 \odot \cdots \odot S^L\right).$$

You can view this as a generalized linear-time $L$-simplicial attention.

One motivation I find useful: “mixture of memories” mechanisms tend to combine multiple memories additively (a router picks or mixes). This proposal combines them multiplicatively, which behaves like feature-wise gating: all memories have to agree, and features can be amplified or suppressed by products.

In practice, the open questions dominate:
- normalization (row-norm vs no row-norm + RMSNorm, or explicit gating),
- whether RoPE-like positional structure still helps or becomes awkward under multiplicative state composition,
- and whether the mechanism learns something meaningfully different than “more projections + gating.”

So far, my results have not been clean enough to recommend this as a drop-in replacement. The main value for me has been as a lens: if the bottleneck is the shared memory $S$, one way to increase expressivity is to increase the *structure* of the memory interaction without blowing up the $N$ dependence.

```python {linenos=false}
import torch

def multilinear_attn(q, ks, vs):
    """
    Multilinear attention: elementwise product of L feature-space memories.

    Each (ks[l], vs[l]) pair contributes one D×D memory state; the states
    are multiplied elementwise before contracting with q.

    Args:
        q:  [B, H, N, D]       queries
        ks: [L, B, H, N, D]    L key projections
        vs: [L, B, H, N, D]    L value projections

    Returns:
        y: [B, H, N, D]    output
    """
    states = ks.mT @ vs          # [L, B, H, D, D]
    state  = states.prod(dim=0)  # [B, H, D, D]
    return q @ state
```

Strassen-style linearized mixing can be viewed as another structured memory composition:

```python {linenos=false}
import torch

def strassen_linear_attn(q, k1, v1, k2, v2, g1, g2, g3, g4, scale=None):
    """
    Strassen-style linearized mixing: structured combination of two memories.

    Args:
        q:     [B, H, N, D]    queries
        k1:    [B, H, N, D]    first key projection
        v1:    [B, H, N, D]    first value projection
        k2:    [B, H, N, D]    second key projection
        v2:    [B, H, N, D]    second value projection
        g1-g4: [B, H, N, D]   learned gate tensors
        scale: float           normalization scale

    Returns:
        y: [B, H, N, D]    output
    """
    if scale is None:
        N = q.size(-2)
        scale = N ** 05

    S1 = (k1.mT / scale) @ (v1 / scale)      # [B, H, D, D]
    S2 = (k2.mT / scale) @ (v2 / scale)      # [B, H, D, D]
    v1_sum = v1.sum(dim=2, keepdim=True)      # [B, H, 1, D]
    v2_sum = v2.sum(dim=2, keepdim=True)      # [B, H, 1, D]

    y1 = (q @ S1) * v2_sum
    y2 = (S1 * S2).sum(dim=-2, keepdim=True).expand_as(q)
    y3 = (q @ S2) * v1_sum
    y4 = q @ (S1 * S2)
    return y1 * g1 + y2 * g2 + y3 * g3 + y4 * g4
```

---

## Proposal: higher-order memory states (triple / quad attention)

A second idea is to increase the capacity of the pooled state itself. Standard linear attention compresses everything into $S\in\mathbb{R}^{D\times D}$. If you believe tasks are bottlenecked by this memory, you can increase its order.

Triple attention uses a $D\times D\times D$ state:

$$S_{ijk} = \sum_{n=1}^N K^1_{ni}V_{nj}K^2_{nk},$$

and

$$Y_{nj} = \sum_{i,k=1}^D Q^1_{ni}S_{ijk}Q^2_{nk}.$$

Quad attention similarly uses a $D\times D\times D\times D$ state:

$$S_{ijk\ell}=\sum_{n=1}^N K^1_{ni}V_{nj}K^2_{nk}K^3_{n\ell},$$

and

$$Y_{nj}=\sum_{i,k,\ell=1}^D Q^1_{ni}S_{ijk\ell}Q^2_{nk}Q^3_{n\ell}.$$

These preserve linear scaling in $N$ but increase polynomial cost in $D$. That makes them plausible only when $D$ is small and kernels are highly optimized. In my own experiments, stability and memory traffic become the main hurdles quickly, so I view these as “maybe useful for specific bottlenecked settings,” not a general recipe.

```python {linenos=false}
import torch

def triple_attn(q1, q2, k1, k2, v):
    """
    Triple attention: third-order feature-space memory, linear in N.

    Args:
        q1: [B, H, N, D]    first query projection
        q2: [B, H, N, D]    second query projection
        k1: [B, H, N, D]    first key projection
        k2: [B, H, N, D]    second key projection
        v:  [B, H, N, D]    values

    Returns:
        y: [B, H, N, D]    output
    """
    state = torch.einsum('bhni,bhnj,bhnk->bhijk', k1, v, k2)   # [B, H, D, D, D]
    return torch.einsum('bhni,bhijk,bhnk->bhnj', q1, state, q2)


def quad_attn(q1, q2, q3, k1, k2, k3, v):
    """
    Quad attention: fourth-order feature-space memory, linear in N.

    Args:
        q1-q3: [B, H, N, D]    query projections
        k1-k3: [B, H, N, D]    key projections
        v:     [B, H, N, D]    values

    Returns:
        y: [B, H, N, D]    output
    """
    state = torch.einsum('bhni,bhnj,bhnk,bhnl->bhijkl', k1, v, k2, k3)  # [B, H, D, D, D, D]
    return torch.einsum('bhni,bhijkl,bhnk,bhnl->bhnj', q1, state, q2, q3)
```

---

## Where low-rank bottlenecks fit (and why linearizing FLARE collapses)

Low-rank methods like FLARE reduce cost by routing through an intermediate set of latent tokens ($M\ll N$). A subtle point is that they rely on a nonlinearity (softmax) that prevents full associativity collapse. If you fully linearize a two-stage gather–scatter mechanism, you typically recover something like $Y \approx K\,(Q^\top Q)\,(K^\top V)$, which is still governed by a feature-space state and no longer depends on $M$ in an interesting way.

I like keeping this sanity check in mind: if a design’s efficiency comes purely from associativity, then without an additional mechanism (nonlinearity, gating, recurrence, or structured bottlenecks) it tends to inherit the same shared-memory limitations.

FLARE gather-scatter:

```python {linenos=false}
import torch.nn.functional as F

def flare_mixer(q, k, v, scale=1.0):
    """
    FLARE gather–scatter: low-rank global mixing via two SDPA calls.

    Args:
        q: [H, M, D]      latent queries (per head, shared across batch)
        k: [B, H, N, D]   token keys
        v: [B, H, N, D]   token values
        scale: float      attention scale

    Returns:
        y: [B, H, N, D]   mixed token outputs
    """
    qb = q.unsqueeze(0)                                          # [1, H, M, D]
    z  = F.scaled_dot_product_attention(qb, k, v, scale=scale)  # [B, H, M, D]
    y  = F.scaled_dot_product_attention(k, qb, z, scale=scale)  # [B, H, N, D]
    return y
```

---

## Practical notes (from my experiments)

A few things that have mattered more than I expected:
- RMSNorm has been more stable than LayerNorm in mixed precision for these variants.
- Normalization placement can dominate behavior. A recurring trick in linear attention is to simplify or remove row-normalization and stabilize later (e.g., RMSNorm after the mixer).
- Hybrid stacks are worth trying. Interleaving low-rank blocks with multilinear or higher-order blocks could be a reasonable way to trade off routing flexibility and stability, even if the standalone higher-order block is finicky.

---

## Benchmarks I care about

I’m mainly interested in whether these mechanisms work beyond toy settings:
- PDE surrogates (steady and transient), where permutation equivariance matters and token ordering tricks are brittle.
- Long Range Arena, as a stress test for long-context sequence modeling.
- Comprehensive Attention Benchmark, to compare operators across a broader task suite.

---

## Takeaways

1. Vanilla linear attention is efficient because of associativity, but that same compression creates a shared-memory bottleneck that weakens token-specific routing.
2. Higher-order attention is a principled direction if you believe pairwise mixing is the wrong inductive bias for certain tasks, but naive formulations are intractable.
3. Multilinear “memories” provide a clean linear-time way to inject higher-order structure, but in my experiments so far they have been hard to stabilize and have not consistently outperformed simpler baselines.
4. If there is a practical path forward, I suspect it will involve careful normalization and gating, and likely hybrid designs that combine low-rank routing with higher-order feature-space interactions.

If I were building the next round of experiments, my default plan would be:
- start with a strong $L=1$ linear baseline (with modern normalization),
- then test $L=2$ multilinear as the cheapest higher-order extension,
- and only pursue triple/quad-state attention if there is clear evidence that the $D\times D$ memory is the dominant bottleneck.

## References

1. Katharopoulos, A. et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML (2020). [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
2. Choromanski, K. et al. *Rethinking Attention with Performers*. ICLR (2021). [https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794)
3. Xiong, R. et al. *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*. AAAI (2021). [https://arxiv.org/abs/2102.03902](https://arxiv.org/abs/2102.03902)
4. Sun, Y. et al. *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv (2023). [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621)
5. Zhang, B., and Sennrich, R. *Root Mean Square Layer Normalization*. NeurIPS (2019). [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
6. Tay, Y. et al. *Long Range Arena: A Benchmark for Efficient Transformers*. ICLR (2021). [https://arxiv.org/abs/2011.04006](https://arxiv.org/abs/2011.04006)
7. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS (2017). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
8. Kozachinskiy, A. et al. *Strassen Attention, Split VC Dimension and Compositionality in Transformers*. arXiv (2025). [https://arxiv.org/abs/2501.19215](https://arxiv.org/abs/2501.19215)
9. Roy, A. et al. *Fast and Simplex: 2-Simplicial Attention in Triton*. arXiv (2025). [https://arxiv.org/abs/2507.02754](https://arxiv.org/abs/2507.02754)
10. Qin, Z. et al. *The Devil in Linear Transformer*. arXiv (2022). [https://arxiv.org/abs/2210.10340](https://arxiv.org/abs/2210.10340)
