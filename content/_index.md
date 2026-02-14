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
  I design transformer architectures with explicit attention to scaling and memory efficiency.
  My recent work, FLARE, enables million-token regimes on a single GPU.
  I implement new architectures directly in PyTorch and Triton.
  My background spans high-performance computing, numerical analysis, and computational fluid dynamics.
  </p>
  <p><a href="https://www.linkedin.com/in/vpuri3/">LinkedIn</a> | <a href="https://github.com/vpuri3">GitHub</a> | <a href="https://scholar.google.com/citations?user=2N-Q4YkAAAAJ">Google Scholar</a> | <a href="mailto:vedantpuri@cmu.edu"><code>vedantpuri@cmu.edu</code></a></p>
</div>

{{< card id="research-interests" >}}
## Research Interests

- Efficient attention architectures
- Numerical methods for ML and for PDEs
- Scientific machine learning
{{< /card >}}

{{< card id="featured-work" >}}
## Featured Work

{{< projectcard >}}
### FLARE - Fast Low-rank Attention Routing Engine
- Derived a flexible low-rank reformulation of self-attention via latent routing
- Reduced quadratic complexity of self-attention to linear complexity while preserving global communication.
- Demonstrated scaling to 1M tokens on a single H100 GPU, attaining over 200x speedup over vanilla self-attention.
- Implemented attention modules in PyTorch and Triton with reproducible scaling experiments.
- Evaluated across PDE surrogate modeling, NLP, and vision benchmarks.

Ongoing work extends FLARE to decoder-only language modeling. This involves implementing Triton kernels for causal attention, including separate prefill and decode paths, and adapting low-rank attention mechanisms for autoregressive training and memory-constrained inference.

[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/FLARE.py) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/FLARE.py?style=social)](https://github.com/vpuri3/FLARE.py/stargazers) | [Huggingface Paper Page](https://huggingface.co/papers/2508.12594)

![FLARE architecture overview](/assets/flare-overview.png)
{{< /projectcard >}}

{{< projectcard >}}
### Hybrid equation-based + data-driven PDE modeling framework
- Introduced smooth neural fields as nonlinear spatial ansatz functions in equation-based reduced-order modeling.
- Retained physics-based Galerkin time evolution while learning expressive low-dimensional representations.
- Attained 200x speedup over full-order simulations in transport-dominated regimes.

[Project page](https://vpuri3.github.io/NeuralROMs.jl/dev/) | [Journal of Comp. Phys. paper](https://www.sciencedirect.com/science/article/pii/S0021999125002402) | [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/NeuralROMs.jl) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/NeuralROMs.jl?style=social)](https://github.com/vpuri3/NeuralROMs.jl/stargazers) | [Slides](https://slides.com/vedantpuri/snf-rom-wccm2024) | [Talk](https://youtu.be/zio-_89DJ0g?si=sDVE1c0xJqzVi8bm)

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

{{< card id="open-source" >}}
## Open Source

{{< projectcard >}}
### FLARE [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/FLARE.py) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/FLARE.py?style=social)](https://github.com/vpuri3/FLARE.py/stargazers)

Fast Low-rank Attention Routing Engine for scalable transformer attention.

{{< /projectcard >}}

{{< projectcard >}}
### mlutils.py [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/mlutils.py) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/mlutils.py?style=social)](https://github.com/vpuri3/mlutils.py/stargazers)

Lightweight PyTorch project template and utility toolkit for ML experiments.
{{< /projectcard >}}

{{< projectcard >}}
### Julia Open Source Tools
#### [SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) [![SciMLOperators.jl](https://img.shields.io/badge/GitHub-SciMLOperators.jl-181717?logo=github)](https://github.com/SciML/SciMLOperators.jl) [![SciMLOperators stars](https://img.shields.io/github/stars/SciML/SciMLOperators.jl?style=social)](https://github.com/SciML/SciMLOperators.jl/stargazers)
Operator abstractions for SciML and PDE workflows

#### [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) [![LinearSolve.jl](https://img.shields.io/badge/GitHub-LinearSolve.jl-181717?logo=github)](https://github.com/SciML/LinearSolve.jl) [![LinearSolve stars](https://img.shields.io/github/stars/SciML/LinearSolve.jl?style=social)](https://github.com/SciML/LinearSolve.jl/stargazers)
Linear solver interface for scientific machine learning


Below is a nonexhaustive list of Julia projects that I have contributed to.
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/OrdinaryDiffEq.jl) [![Stars](https://img.shields.io/github/stars/SciML/OrdinaryDiffEq.jl?style=social)](https://github.com/SciML/OrdinaryDiffEq.jl/stargazers)
- [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/NonlinearSolve.jl) [![Stars](https://img.shields.io/github/stars/SciML/NonlinearSolve.jl?style=social)](https://github.com/SciML/NonlinearSolve.jl/stargazers)
- [Optimization.jl](https://github.com/SciML/Optimization.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/Optimization.jl) [![Stars](https://img.shields.io/github/stars/SciML/Optimization.jl?style=social)](https://github.com/SciML/Optimization.jl/stargazers)
- [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/SciMLBase.jl) [![Stars](https://img.shields.io/github/stars/SciML/SciMLBase.jl?style=social)](https://github.com/SciML/SciMLBase.jl/stargazers)
- [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/SciMLSensitivity.jl) [![Stars](https://img.shields.io/github/stars/SciML/SciMLSensitivity.jl?style=social)](https://github.com/SciML/SciMLSensitivity.jl/stargazers)
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/DiffEqFlux.jl) [![Stars](https://img.shields.io/github/stars/SciML/DiffEqFlux.jl?style=social)](https://github.com/SciML/DiffEqFlux.jl/stargazers)
- [StochasticDiffEq.jl](https://github.com/SciML/StochasticDiffEq.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/StochasticDiffEq.jl) [![Stars](https://img.shields.io/github/stars/SciML/StochasticDiffEq.jl?style=social)](https://github.com/SciML/StochasticDiffEq.jl/stargazers)
- [DiffEqBase.jl](https://github.com/SciML/DiffEqBase.jl) [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/SciML/DiffEqBase.jl) [![Stars](https://img.shields.io/github/stars/SciML/DiffEqBase.jl?style=social)](https://github.com/SciML/DiffEqBase.jl/stargazers)
{{< /projectcard >}}

{{< projectcard >}}
### KolmogorovArnold.jl [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/KolmogorovArnold.jl) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/KolmogorovArnold.jl?style=social)](https://github.com/vpuri3/KolmogorovArnold.jl/stargazers)

Julia implementation of Kolmogorov-Arnold Networks with custom gradients for faster training.

{{< /projectcard >}}

{{< projectcard >}}
### FastDiffusion.py [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/FastDiffusion.py) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/FastDiffusion.py?style=social)](https://github.com/vpuri3/FastDiffusion.py/stargazers)

Experiment with trigonometric noise schedule in context of few step diffusion.

{{< /projectcard >}}

{{< projectcard >}}
### NekTools [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/vpuri3/NekTools) [![GitHub stars](https://img.shields.io/github/stars/vpuri3/NekTools?style=social)](https://github.com/vpuri3/NekTools/stargazers)

FORTRAN 77 utilities for turbulence statistics and post-processing in NEK5000.

{{< /projectcard >}}
{{< /card >}}
