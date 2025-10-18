#### FID evaluations
A hierarchical of 2 VAEs + VampPrior:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 4.631768
- FID-Y (analytic true in 𝒴 vs. generated CFG):   2.059666

An Elucidating Diffusion Model with me implementing ODE sampler incorrectly:
EDM:
- FID-Y (analytic true in 𝒴 vs. generated unsup): 1.690798
- FID-Y (analytic true in 𝒴 vs. generated CFG):   0.604588

I made the mistake intentionally to showcase that in term of unreliable indicator: loss is on top of the list, then FID. In my opinion, the best way to visualise generative model's performance is to create a difficult toy example data distribution and see how well any model matching it
