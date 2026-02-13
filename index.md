@def title = "Vedant Puri"
@def tags = ["research", "ml", "scientific-computing"]

# Vedant Puri
PhD Candidate, Carnegie Mellon University  
Efficient Transformer Architectures | Scientific Machine Learning

I design scalable transformer architectures grounded in numerical methods.  
My recent work introduces FLARE, a unified low-rank attention mechanism that scales to million-token problems on a single GPU.  
My background spans HPC, PDE solvers, and scientific computing.

## Featured Work

### FLARE - Fast Low-Rank Attention Routing Engine
- Unified low-rank reformulation of self-attention
- O(NM) memory scaling
- Scales to 1M tokens on a single GPU
- Benchmarked on PDE, NLP, and vision tasks

[Paper](https://huggingface.co/papers/2508.12594) | [Code](https://github.com/vpuri3/FLARE.py) | [arXiv](http://arxiv.org/abs/2508.12594)

![](https://raw.githubusercontent.com/vpuri3/FLARE.py/master/figs/time_memory_bwd.png)

### SNF-ROM
SNF-ROM is a projection-based nonlinear reduced-order model using smooth neural fields for advection-dominated PDEs.

[Journal of Computational Physics paper](https://arxiv.org/abs/2405.14890) | [Code](https://github.com/vpuri3/NeuralROMs.jl)

## Research Themes
- Efficient attention and low-rank transformers
- Neural operators and PDE surrogates
- Numerical methods for ML architectures
- Scientific computing at scale

## Open Source

### FLARE
[FLARE.py](https://github.com/vpuri3/FLARE.py): Fast Low-rank Attention Routing Engine for scalable transformer attention.

### Julia Open Source Tools
- [SciMLOperators.jl](https://github.com/vpuri3/SciMLOperators.jl): operator abstractions for SciML and PDE workflows
- [LinearSolve.jl](https://github.com/vpuri3/LinearSolve.jl): linear solver interface for scientific machine learning

Additional Julia repos I have worked on include [OrdinaryDiffEq.jl](https://github.com/vpuri3/OrdinaryDiffEq.jl), [NonlinearSolve.jl](https://github.com/vpuri3/NonlinearSolve.jl), [Optimization.jl](https://github.com/vpuri3/Optimization.jl), [SciMLBase.jl](https://github.com/vpuri3/SciMLBase.jl), [SciMLSensitivity.jl](https://github.com/vpuri3/SciMLSensitivity.jl), [DiffEqFlux.jl](https://github.com/vpuri3/DiffEqFlux.jl), [StochasticDiffEq.jl](https://github.com/vpuri3/StochasticDiffEq.jl), and [DiffEqBase.jl](https://github.com/vpuri3/DiffEqBase.jl).

### KolmogorovArnold.jl
[KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl): Julia implementation of Kolmogorov-Arnold Networks with custom gradients for faster training.

### NekTools
[NekTools](https://github.com/vpuri3/NekTools): FORTRAN 77 utilities for turbulence statistics and post-processing in NEK5000.

## Blog
Long-form background, older project descriptions, and non-ML writing are archived here:
- [Work archive](/work/)
- [Thoughts archive](/thoughts/)

Planned technical post:
- How to scale attention to 1M tokens on a single GPU
