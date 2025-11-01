My bet is that Stochastic Interpolant would be the best one for predicting joint probability, but let's see.


## Overall commentary
1. It's immediately clear that FID and training loss don't reflect actual fidelity, these are expected
2. What is not immediately clear until I test different gen models is that the effect of adding on guidance term. There are some hints from subsection 8.8: Regularisation in ``SC4/SM8 Advanced Topics in Statistical Machine Learning`` by Tom Rainforth that guidance terms are regularisers. But it's uncertain until I test it here, where it's clearer that because VAE loss function (ELBO loss) doesn't provide enough intrinsic variance, thus the model cannot handle the addon bias from classifier-free guidance, thus it underfits. ``Guiding a Diffusion Model with a Bad Version of Itself`` also makes a remark on CFG making this phenomenon, but the simplest way to phrase this is: too much bias in bias-variance tradeoff, thus the model underfit. To verify my claim, I induce underfitting to Diffusion model in an ablation test, where I intentionally lower training samples or lower sampling steps because common senses say that at least 1 approach guarantee will lead to underfitting. And since the simulation result looks like VAE's result with guidance, this verifies my claim that ``low fidelity loss function doesn't have enough intrinsic variance to take on extra bias from guidance`` 


## Overall objective
Basically emulate this ground truth data distribution perfectly in unsupervised setting: <br>
![image](https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/toy-DataDistribution_unsup.png)

and in semi-supervised setting: <br>
![image](https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/toy-DataDistribution_semisup.png)

