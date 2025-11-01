## Brief introduction of different variations:
Classical VAE by Kingma 2013 is essentially a hierarchical of 1 VAE with normal distribution as prior. Here, I use a hierarchical of 2 VAEs, as illustrated by figure 1b in ``VAE with VampPrior`` paper. Then I use different mixture distribution as prior (normalPrior), such as Mixture of Gaussians (MogPrior) or Mixture of Variational Posteriors (VampPrior)

## FID evaluations
A hierarchical of 2 VAEs + VampPrior:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 602.899663
- FID-Y (analytic true in 𝒴 vs. generated CFG):   0.379020

A hierarchical of 2 VAEs + MogPrior:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 2.701265
- FID-Y (analytic true in 𝒴 vs. generated CFG):   1.722763

A hierarchical of 2 VAEs + normalPrior:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 2.264380

- FID-Y (analytic true in 𝒴 vs. generated CFG):   1.693497
