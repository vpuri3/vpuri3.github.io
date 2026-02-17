+++
title = "Our job as computational engineers"
date = 2022-01-01T00:00:00-00:00
draft = false
description = "Collected long-form notes and technical reflections."
+++

## Why math is interesting

The foundation of my interest in math was laid in my freshman year when I was appointed course staff to a sophomore engineering course. As I taught students to model mechanical interactions using forces and moments, I learned not only to communicate my ideas to newcomers in engineering but also how to challenge my own preconceived notions. Explaining why a certain unseen force has to exist to maintain equilibrium is like talking in a different language, in the sense that certain things are obvious (evident enough not to require proving) to me but not to a student. I have to explain that concentrated point forces, reaction moments and all these unseen entities are fictitious objects made by engineers and scientists to model and approximate reality.

I could relate to students as I was in the same situation mere months ago, asking the same questions in disbelief. However, as I rolled concepts around in my head and thought things over I realized that however obscure, these seemingly unreal objects actually exist and their approximations are appropriate. In fact, their presence makes perfect physical sense. This epiphany, a shining moment of clarity is a reward in itself - one that makes you not want to stop thinking. Such daunting thought exercises attract me further towards the theoretical nature of engineering. It is truly marvelous to me that this edifice of human knowledge is derived from, and can be parsed through with, logical deductions based on a combination of observations and axiomatic thinking.

In teaching foundational topics, including proving the validity of the variational principles of mechanics, I articulated my own learning style. While studying discretization techniques in numerics courses, I became interested in the underlying variational formulation and the analytical treatment of PDEs. I studied Sobolev spaces, and wrote my own set of reference notes on the topic. I enjoyed the analytical rigor and saw a second degree in mathematics as means to place my engineering education in context within applied mathematics.

## Writing Proofs

Part of writing proofs in mathematics is convincing yourself of the reality of the statement, say P. That given the nature of reality, P is the only way things can go, and no other way is feasible.

A way to convince yourself is to "stare at it until you believe it is true." The methods sounds passive but in reality is fairly involved. While staring, I run thought experiments in my head imagining all sorts of test functions, sets, and what not -- and wonder how they interact with with the statement P. Over time the visualizations of P in your head become intricate and you become more comfortable with the idea. The aim is to get an exhaustive idea of the nature of P and be able to say with confidence "how could it be anything but P!"

Now that the act of self delusion is over, the next (harder) step is to attack your supposition from every angle to see how much of it is true.

## Deep learning, composition, gradient flows

In my work at Carnegie Mellon University, we have been able to recover finite element function spaces using hand-crafted neural network architectures, implying that the space of functions that can be represented by neural networks encompasses the space of finite elements. The integration of gradient flows through differentiable programming therefore allow us to attack high dimensional problems nested within engineering workflows such as automated meshing, a significant bottleneck in engineering workflows.

Deep learning derives its powers from the composition of nonlienar operations, a setup in which model parameters can be tuned via gradient descent. It's an interesting problem: integrating gradient flows in state of the art computational technology integrations of gradient flows (good for high dimensional problems) with state of the art simulation technology (good for 2D/3D).

Need to understand (Barron space ideas) what function space outputs of neural networks lie in, and what are appropriate norms for analysis in these spaces. How to manipulate them to get a desired output? how to effectively do regression?

## Our job as computational engineers

Computational workflows form a tall stack of abstractions that map physical quantities like velocity to distributed data structures acted upon by numerical operators. Scientific computing consolidates my interest in physics and the intricate mathematical framework that attempts to explain it. As such, developing mathematical models to accurately capture physics is a creative problem I like spending time on.

Present challenges in predictive modelling vary with the level of analysis applied from robust meshing of complex geometries to reliable turbulence models. With the aim of optimizing Computer Aided Engineering workflows, my focus is on developing application-specific tools for problems that advance humanity's technological prowess.

Differentiable programming offer a superior way of integrating data with theory. Our job is to develop the tools that further our understanding the most. Aim is to develop problem specific architectures for PDE solving in complex geometries. The task is to develop a hybrid (sort) system where we can utilizes gradient flows (DL) and high order function approximation (FEM type) in an efficient way.

## Misc (do not read)

This seems to be the nature of the universe: no matter how far you zoom in, you can still zoom in more. we're just limited by our abilities not but a lack of detail to discover. no matter how far you zoom out, you can zoom out more. there's always more stuff to look at. let's call the un-computable transcendental. now by virtue of its infiniteness and by virtue of the the finiteness of human abilities of logic and to reason, the relationship one has with the nature of the universe has to be private because nobody can articulate it completely and explain it to themselves or to others. it's like an infinitely layered story. something you can go on and on about without getting to the bottom of it. like a great book. or a great story. it's by definition that the transcendent cannot be reduced a single or any number of thoughts or ideas or things without it automatically losing its essence. Ok I should get back to work.

## Tangent on tet meshes

Tetrehedral meshes are everywhere. Geometry is designed with splines and immidiately translated and shared with tet meshes. Splines rule CAD because knots give a designer granular control over the shape of their part.

I'm very curious about implicit geometry representations, particularly high order spectral representations (neural too because that's just a different basis to same difference) because it provides a bigger bang for your buck: you can accurately represent a more complicated function with the same number of points with chebychev distribution rather than uniform distribution (Runge phenomena). The issue with spectral geometry processing might be with patching, ie how to make different spectral pieces talk to each other. In Spectral Element Method, C0 continuity is handled via a gather-scatter operation, meaning that the adjacency matrix only needs to account for the boundary nodes.

It would be interesting to learn how splines preserve locality, and how they impose cotinuity conditions across "elements".

## References

1. Brenner, S. C., and Scott, L. R. *The Mathematical Theory of Finite Element Methods* (3rd ed.). Springer (2008). [https://doi.org/10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. Evans, L. C. *Partial Differential Equations* (2nd ed.). AMS (2010). [https://bookstore.ams.org/gsm-19-r](https://bookstore.ams.org/gsm-19-r)
3. Adams, R. A., and Fournier, J. J. F. *Sobolev Spaces* (2nd ed.). Academic Press (2003). [https://shop.elsevier.com/books/sobolev-spaces/adams/978-0-12-044143-3](https://shop.elsevier.com/books/sobolev-spaces/adams/978-0-12-044143-3)
4. Quarteroni, A., and Rozza, G. (eds.). *Reduced Order Methods for Modeling and Computational Reduction*. Springer (2014). [https://doi.org/10.1007/978-3-319-02090-7](https://doi.org/10.1007/978-3-319-02090-7)
5. Hughes, T. J. R., Cottrell, J. A., and Bazilevs, Y. *Isogeometric Analysis: CAD, Finite Elements, NURBS, Exact Geometry and Mesh Refinement*. CMAME (2005). [https://doi.org/10.1016/j.cma.2004.10.008](https://doi.org/10.1016/j.cma.2004.10.008)
