There are A LOT of unique variants of Diffusion so here I only pick the most common one, and leave other variants to be dealt with later when I discuss about Stochastic Interpolant. The main takeaway is simple, the common myth Diffusion has to couple with OU processs (i.e. reverse time SDE) is wrong. With reverse ODE, it does just nearly just as good, not just because the FID says so also because the actual simulation result says so! 

## FID evaluations
EDM-SDE (1024 sampling steps, 2047 NFE):
- FID-Y (analytic true in 𝒴 vs. generated unsup): 0.976439
- FID-Y (analytic true in 𝒴 vs. generated CFG):   1.230182

EDM-ODE (1024 sampling steps, 2047 NFE):
- FID-Y (analytic true in 𝒴 vs. generated unsup): 0.966004
- FID-Y (analytic true in 𝒴 vs. generated CFG):   1.301372


## Ablation test 1: lower number of sampling steps
EDM-ODE (100 sampling steps, 199 NFE): 
- FID-Y (analytic true in 𝒴 vs. generated unsup): 0.962172
- FID-Y (analytic true in 𝒴 vs. generated CFG):   1.312513

EDM-ODE (10 sampling steps, 19 NFE): 
- FID-Y (analytic true in 𝒴 vs. generated unsup): 0.558943
- FID-Y (analytic true in 𝒴 vs. generated CFG):   2.587558

EDM-ODE (1 sampling steps, 1 NFE):
- FID-Y (analytic true in 𝒴 vs. generated unsup): 5.411988
- FID-Y (analytic true in 𝒴 vs. generated CFG):   0.652704


## Ablation test 2: lower number of training samples
EDM-ODE, 1000 training samples:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 5.404045
- FID-Y (analytic true in 𝒴 vs. generated CFG):   2.908479

EDM-ODE, 500 training samples:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 5.626093
- FID-Y (analytic true in 𝒴 vs. generated CFG):   4.897393

EDM-ODE, 100 training samples:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 5.693155
- FID-Y (analytic true in 𝒴 vs. generated CFG):   5.343436
