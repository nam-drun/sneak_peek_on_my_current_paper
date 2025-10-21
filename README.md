I hope you are here after reading the Abstract of my paper, if not please read it at [my linkedin post](https://www.linkedin.com/posts/activity-7384852062439800832-C9Zu?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC2h3ncBpRfbKTT1DIVOja9SV5bC4ODmJIA) first - and if your link is broken, it's because I set my post to only show to people in my connection (it's just my precaution). Anyway yes, I do all these in solo research. Still, I'm grateful to get feedbacks from postdoc and phd people occasionally to keep my paper compact and concise. Here are what I intend to say:

#### Priority when designing generative model (bottom means higher priority otherwise the whole system implodes)
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/generative-model-design-priority.png" width="400" height="400">


#### Then I talk about Stochastic Interpolant
To keep it digestible to beginner, I spend Background section to cover something that we see in everyday commercial life like histogram plot and density plot all the way to Stochastic Interpolant. Then to make a point that Stochastic Interpolant enables trivial integrating an innovation from other generative models. I pick a good intuition that have been consistently attempted in VAE and Diffusion but failed to capitalise, then try it on Stochastic Interpolant and use this framework to identify what went wrong when the intuition was attempted on VAE and Diffusion<br>
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/20251018_094648.jpg" width="400" height="400">


#### My experiment procedure is:
(i) Create a toy data distribution by manually parameterising Mixture-of-Gaussian since it has a universal density approximation property like deep neural network <br>
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/toy-DataDistribution_unsup-guidance.png" width="400" height="400">

Note, all models in my paper are forced to work with only 1000 training samples and only allowed to use 2 blocks-MLP with SwigLU activation function as backbone. **The emphasis is see how far can loss function and optimisation algorithm and sampling algorithm and guidance together can push the model's generative performance** <br>
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/what%20model%20sees%20with%20training%20samples.png" width="400" height="400">

(ii) Then make an arbitrary generative model (i.e. Stochastic Interpolant, Flow, Diffusion, VAE) approximate this toy data distribution. I also plot training loss and probabilistic evaluation like FID to showcase that they are **unreliable indicator**. Here's a thought experiment: <br>
This is from Diffusion: <br>
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/denoisingMatchingLoss_for_2trainings.png" width="400" height="400">

This is from hVAE + VampPrior: <br>
<img src="https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/elboLoss_for_2_trainings.png" width="400" height="400">

Which one do you think approximate data distribution better? <br>

...<br>
After making your guess, see the answer in VAEvsDiffusion folder
