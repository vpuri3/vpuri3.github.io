+++
title = "Home"
+++

<div class="hero-card">
  <div class="hero-top">
    <img class="hero-photo" src="/assets/profile-headshot.jpg" alt="Vedant Puri profile photo">
    <div class="hero-copy">
      <h1>Vedant Puri</h1>
      <p class="hero-subtitle">PhD Candidate, Carnegie Mellon University</p>
      <p class="hero-tagline">Efficient Transformer Architectures | Scientific Machine Learning</p>
    </div>
  </div>
  <p>
  I design scalable transformer architectures grounded in numerical methods.
  My recent work introduces FLARE, a unified low-rank attention mechanism that scales to million-token problems on a single GPU.
  My background spans high performance computing, partial differential equation solvers, and scientific machine learning.
  </p>
  <p><a href="https://www.linkedin.com/in/vpuri3/">LinkedIn</a> | <a href="https://github.com/vpuri3">GitHub</a> | <a href="https://scholar.google.com/citations?user=2N-Q4YkAAAAJ">Google Scholar</a></p>
</div>

{{< card id="featured-work" >}}
## Featured Work

{{< projectcard >}}
### FLARE - Fast Low-Rank Attention Routing Engine
- Unified low-rank reformulation of self-attention
- O(NM) memory scaling
- Scales to 1M tokens on a single GPU
- Benchmarked on PDE, NLP, and vision tasks

[Paper](https://huggingface.co/papers/2508.12594) | [Code](https://github.com/vpuri3/FLARE.py) | [arXiv](http://arxiv.org/abs/2508.12594)

![FLARE architecture overview](/assets/flare-overview.png)
{{< /projectcard >}}

{{< projectcard >}}
### SNF-ROM
SNF-ROM is a projection-based nonlinear reduced-order modeling framework with smooth neural fields for advection-dominated PDEs.

- Combines projection-based ROM with continuous neural field representations
- Targets challenging transport-dominated PDE regimes
- Implemented in Julia with experiment suites for 1D and 2D advection and Burgers systems
- Includes reproducible pipelines for dataset generation, training, and model comparison

[Project page](https://vpuri3.github.io/NeuralROMs.jl/dev/) | [JCP paper](https://arxiv.org/abs/2405.14890) | [Code](https://github.com/vpuri3/NeuralROMs.jl) | [Slides](https://slides.com/vedantpuri/snf-rom-wccm2024) | [Talk](https://youtu.be/zio-_89DJ0g?si=sDVE1c0xJqzVi8bm)

![SNF-ROM online stage architecture](/assets/snfrom-online-stage.png)
{{< /projectcard >}}
{{< /card >}}

{{< card id="previous-work" >}}
## Previous Work: Computational fluid dynamics on HPC systems

{{< projectcard >}}
I previously worked on turbulence simulation and analysis workflows in high-performance computing settings, with emphasis on spectral element methods and large-scale post-processing.
This background in numerical methods and PDE solvers informs how I design stable and efficient transformer architectures for scientific ML.

![Velocity magnitude for flow past wall-mounted cube](/assets/wall-mounted-cube-velocity.jpg)
*Velocity magnitude for flow past wall-mounted cube case at Reynolds Number 3900 with respect to cube height. Computation performed using spectral element code NEK5000 at Argonne Leadership Computing Facility.*
{{< /projectcard >}}
{{< /card >}}

{{< card id="photography" >}}
## Not Work

{{< projectcard >}}
### Not So Up-to-Date Photography Portfolio
For the past decade, I have used a Canon DSLR as an excuse to walk around and photograph people, geometry, and city texture.

[Open portfolio page](/photography/) | [Flickr](https://www.flickr.com/photos/128280868@N05/)
{{< /projectcard >}}
{{< /card >}}

{{< card id="research-themes" >}}
## Research Themes

- Efficient attention and low-rank transformers
- Neural operators and PDE surrogates
- Numerical methods for ML architectures
- Scientific computing at scale
{{< /card >}}

{{< card id="open-source" >}}
## Open Source

{{< projectcard >}}
### FLARE
[FLARE.py](https://github.com/vpuri3/FLARE.py): Fast Low-rank Attention Routing Engine for scalable transformer attention.
{{< /projectcard >}}

{{< projectcard >}}
### Julia Open Source Tools
- [SciMLOperators.jl](https://github.com/vpuri3/SciMLOperators.jl): operator abstractions for SciML and PDE workflows
- [LinearSolve.jl](https://github.com/vpuri3/LinearSolve.jl): linear solver interface for scientific machine learning

Additional Julia repos I have worked on include [OrdinaryDiffEq.jl](https://github.com/vpuri3/OrdinaryDiffEq.jl), [NonlinearSolve.jl](https://github.com/vpuri3/NonlinearSolve.jl), [Optimization.jl](https://github.com/vpuri3/Optimization.jl), [SciMLBase.jl](https://github.com/vpuri3/SciMLBase.jl), [SciMLSensitivity.jl](https://github.com/vpuri3/SciMLSensitivity.jl), [DiffEqFlux.jl](https://github.com/vpuri3/DiffEqFlux.jl), [StochasticDiffEq.jl](https://github.com/vpuri3/StochasticDiffEq.jl), and [DiffEqBase.jl](https://github.com/vpuri3/DiffEqBase.jl).
{{< /projectcard >}}

{{< projectcard >}}
### KolmogorovArnold.jl
[KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl): Julia implementation of Kolmogorov-Arnold Networks with custom gradients for faster training.
{{< /projectcard >}}

{{< projectcard >}}
### NekTools
[NekTools](https://github.com/vpuri3/NekTools): FORTRAN 77 utilities for turbulence statistics and post-processing in NEK5000.
{{< /projectcard >}}
{{< /card >}}
