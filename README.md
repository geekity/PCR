PCR
===

Basic Parallel Cyclic Reduction GPU Solver, as based on:

[1] Y. Zhang, J. Cohen, A. A. Davidson and J. D. Owens, A Hybrid Method for Solving Tridiagonal Systems on the GPU, Chapter 11, GPU Computing Gems Jade Edition
[2] Z. Wei, B. Jang, Y. Zhang and Y. Jia, Parallelizing Alternating Direction Implicit Solver on GPU, Procedia Computer Science 18 (2013) 389-398

This method implements somewhat greater generalisation of the equation system size than in Zhang et al. [1], where this is now not constrained by hardware thread limitations. This is achieved through implementing method (b) from section 4.2.1 of Wei et al. [2]. At the same time we lose the convenient memory optimisation of using shared memory for the reduction. 

This solver comes with absolutely no guarantees but feel free to let me know when/if stuff breaks.
