+++
title = "Work"
+++

{{< card >}}


# Work summary
This page is an extended curriculum viatte with unnecessary commentary. Check out my single page [CV](https://github.com/vpuri3/vpCV/raw/master/vpCV.pdf) if you're feeling lazy.


{{< projectcard >}}
## BS @ University of Illinois Urbana-Champaign
I attended University of Illinois Urbana-Champaign from 2015 to 2019, earning Bachelor's degrees in *Mathematics*, *Engineering Mechanics*, and a minor in *Computational Science and Engineering*. While at university, I became interested in the numerical treatment of partial differential equations for physical modeling. To that effect, I made an attempt to comprehend the entire knowledge stack of computer simulations: I studied Hamiltonian mechanics to learn how physics is modeled; Sobolev spaces to learn the mathematical treatment of differential operators; and numerical algorithms for converting continuous equationsinto solvable problems. 

To complement my classwork, I wrote [spectral element codes](https://github.com/vpuri3/spec) to solve fluid-flows in interesting geometries, and interned first at the National Center for Supercomputing Applications on numerical PDEs, and then at the Argonne National Laboratory on wall-bounded turbulent flows. The work from ANL resulted in my senior thesis. I also have substantial on-campus engagement through Society for Engineering Mechanics, a student organization that I led in my senior year.

### Society for Engineering Mechanics
We engineerined mechanics. Add link to projects like chocolate 3D printer, SIIP work.

### *Introductory Statics* Course Assistant
At UIUC, I discovered an interest in teaching when I was selected in freshman year to be a course assistant for *Introductory Statics*, a mechanics course where students learn to model mechanical interactions as forces and moments. The instructors for *Statics* invited me to attend the *Strategic Instructional Innovations Program* meetings to assist with curriculum design for mechanics courses serving $2500$ students annually. I created engineering-design oriented activities for students and coordinated with the Society for Engineering Mechanics, a student organization that I led in my senior year, to manufacture instructional demonstrations for introductory courses in *Statics*, *Dynamics*, and *Solid Mechanics*.

### Illini Hyperloop
As our capstone project at UIUC, my group implemented a passive cooling solution capable of absorbing up to 300 kJ of heat. The solution would be used for the propulsion system of the Hyperloop pod of Illini Hyperloop team participating in SpaceX Hyperloop Competition 2019. Fabrication was handled by the project sponsor, Novark Technologies, Inc. The project video, report, and summary can be found [here](https://github.com/vpuri3/IlliniHyperloop).

{{< /projectcard >}}

{{< projectcard >}}
## National Center for Supercomputing Applications, 2017-18

My work at NCSA on initial data generation for gravitational wave simulations resulted in a talk that I delivered at the *American Physical Society April Meeting 2018*. Under Dr. Roland Haas of the Gravity Group, I solved nonlinear elliptic initial data equations for gravitational wave simulations in the cosmological framework Einstein Toolkit. I implemented the novel Scheduled Relaxation Jacobi technique which relies on precomputing a set of relaxation factors by nonlinear optimization for Laplacian-dominated PDEs. The relaxation solve was embedded within a Newton-Raphson loop to alleviate the stiffness due to nonlinearity, reducing the time-to-solution by several orders of magnitude.

{{< /projectcard >}}

{{< projectcard >}}
## Argonne Natinoal Laboratory, 2018
In the summer of 2018, I studied wall-bounded turbulence flows at Argonne Natinoal Lab Dr. Ramesh Balakrishnan, and Dr. Aleksandr Obabko of the Mathematics and Computer Science divisio. The project was part of a larger United States Department of Energy effort on simulating airflow in open-ocean windfarms. I examined the physics of flows over curved geometries by conducting Direct Numerical Simulations.

To facilitate the development of wall-models for Large Eddy Simulations, I computed the terms of the tensor Reynolds Stress Transport Equation within the spectral element fluid dynamics code NEK5000. The converged Reynolds Stress budgets (third order statistics) proper resolution of the high-frequency modes in my calculation, and reveal the spatial distribution and relative magnitudes of the production, dissipation, and transport mechanisms of turbulent energy. My FORTRAN 77 routines for turbulence statistics computation and post-processing in NEK5000 can be found in this [git](https://github.com/vpuri3/NekTools) repo.

As the scope of the project was too broad for one summer, I continued the work as thesis research at University of Illinois.

{{< /projectcard >}}

{{< projectcard >}}
## Argonne Natinoal Laboratory, 2020
I returned to Argonne after graduation to continue working on turbulence modeling problems, with a focus on complex geometry, and hierarchical simulations where the output of a highly resolved but simplified calculation would be used in modelling unresolved physics in a larger more complicated simulation. I conducted Large Eddy Simulations (LES) and Reynolds Averaged Navier-Stokes (RANS) calculations of airflow near building-like geometries for the DOE’s Distributed Windproject. The overarching goal of the project is to obtain high-resolution  LES  and  DNS  data to  support  development  of subgrid-stress  models  and  wall-models for Detached Eddy Simulations and Reynolds Averaged Navier-Stokes simulations.

{{< /projectcard >}}

{{< projectcard >}}
## Carnegie Mellon University, 2020
In July 2020, I connected with Professor Venkat Viswanathan at Carnegie Mellon University, who expressed interest in developing differentiable programming models for computational fluid dynamics problems. With his group, I tackled problems in scientific computing in a very broad sense, while pushing out packages in the Julia ecosystem. In our work, we have been able to recover finite element function spaces using hand-crafted neural network architectures, implying that the space of functions that can be represented by neural networks encompasses the space of finite elements.

As such, the Viswanathan research group's emphasis on pushing out open source packages and ties to commercial markets would constrain and guide my research endeavors in a fruitful direction.

Started work on fully differentiable spectral element solver implemented in Julia.

{{< /projectcard >}}

{{< projectcard >}}
## Julia Computing, 2021
I am working on deploying Physics Informed Neural Networks for Julia Computing's commercial products under Dr. Chris Rackauckas. I also wrote the linear solver interface for `DifferentialEquations.jl` ecosystem.

# PhD @ Carnegie Mellon University, 2022-

In January 2022, I will join Carneige Mellon University to continue working on numerical PDEs with the [Viswanathan Research Group](https://www.cmu.edu/me/venkatgroup/) in the Mechanical Engineering Department. Having worked with the group for about a year, the group's emplasis on pushing out open source packages and ties to commercial markets constrains and guides my research endeavors in a fruitful direction. 

{{< /projectcard >}}

{{< /card >}}
