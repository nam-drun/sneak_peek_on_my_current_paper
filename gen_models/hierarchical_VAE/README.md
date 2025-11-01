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

## Overall commentary
1. It's immediately clear that FID and training loss don't reflect actual fidelity, these are expected
2. What is not immediately clear until I test different gen models is that the effect of adding on guidance term. There are some hints from subsection 8.8: Regularisation in ``SC4/SM8 Advanced Topics in Statistical Machine Learning`` by Tom Rainforth that guidance terms are regularisers. But it's uncertain until I test it here, where it's clearer that because VAE loss function (ELBO loss) doesn't provide enough intrinsic variance, thus the model cannot handle the addon bias from classifier-free guidance, thus it underfits. ``Guiding a Diffusion Model with a Bad Version of Itself`` also makes a remark on CFG making this phenomenon, but the simplest way to phrase this is: too much bias in bias-variance tradeoff, thus the model underfit

