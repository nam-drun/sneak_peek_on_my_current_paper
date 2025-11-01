A sure way to induce underfitting. By the way, what's surprising is that even though I use 1/50 training dataset size, Diffusion is very robust. It wasn't until I literally only use 10 training sample before the model starts to underfit

## Unsupervised setting
![image](https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/gen_models/Diffusion/EDM-ODE%20ablation%20test%20-%20low%20training%20samples/100%20samples/unsup_approxDataDistribution.png) 

## Semisupervised setting
![image](https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/gen_models/Diffusion/EDM-ODE%20ablation%20test%20-%20low%20training%20samples/100%20samples/unsup-cfg_approxDataDistribution.png)

## Training Loss for denoising matching loss function
![image](https://github.com/nam-drun/sneak_peek_on_my_current_paper/blob/main/gen_models/Diffusion/EDM-ODE%20ablation%20test%20-%20low%20training%20samples/100%20samples/denoisingMatchingLoss_for_2trainings.png)
