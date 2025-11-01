"""
Stochastic Interpolant (SI)–aligned notation for VAE + Mixture-of-Gaussians (MoG) prior training
========================================================================

We *use* x₀ ~ ρ₀ ⊂ 𝒳 = ℝᵈ (data endpoint in SI) synthesized by `toy3_si.py` (this file imports from it).
We *learn* a VAE with encoder q_φ(z|x), decoder p_θ(x|z), and a Mixture-of-Gaussians prior (alternative to VampPrior):

    p_λ(z) = (1/K) ∑_{k=1}^K N(z; μ_k, diag(σ²_k))                                   [Use Mixture-of-Gaussian as alternative prior to VampPrior]

SI mapping used in this file
----------------------------
- x₀ ∈ 𝒳 is the observed data point (t=0 endpoint). The *noise* endpoint x₁ and interpolation Γ are not used here.
- ρ₀ is the data distribution in 𝒳. We z-score x by μ[ρ₀], σ[ρ₀] (computed from mixture parameters) for training stability.
- U, b are the linear transport parameters such that x₀ = U y + b with y ~ ρ_Y on 𝒴=ℝ² (see generator).
- Symbols:
    d          := ambient dimension (was d)
    mu_0, L_0_lowrank, w_0 := component params of ρ₀ (pushforward of ρ_Y)
    mu_rho0, std_rho0      := mixture mean and per-dim stdev of ρ₀
    sample_x0_unlabeled / sample_x0_labeled := dataset samplers

VampPrior implementation details
--------------------------------
Given encoder outputs μ_φ(x), logσ²_φ(x), we evaluate q_φ(z|u_k) at learnable pseudo-inputs u_k (in data space) and
compute log p_λ(z) via log-sum-exp over components.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
# -----------------------
# LR schedule (warmup + linear decay): adapt from Flow Matching paper
# change number of warm up steps because we use literal batch size instead of effective batch size
# -----------------------
LR_MIN = 1e-08
LR_MAX = 8e-05
WARMUP_STEPS = 261

def lr_warmup_linear_decay(step: int, total_steps: int, warmup_steps: int = WARMUP_STEPS,
                           lr_min: float = LR_MIN, lr_max: float = LR_MAX) -> float:
    """Linear warm-up from lr_min to lr_max over warmup_steps, then
    linear decay from lr_max to lr_min until total_steps."""
    # Guard against degenerate cases
    step = int(step)
    if total_steps <= 0:
        return lr_max
    # Warm-up
    if step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return lr_min + (lr_max - lr_min) * alpha
    # Decay
    rem = total_steps - warmup_steps
    if rem <= 0:
        return lr_max
    beta = (min(step, total_steps) - warmup_steps) / rem
    return lr_max + (lr_min - lr_max) * beta


# --- Import the toy distribution & pushforward machinery (X = T#Y) ---
#from create_ToyExample_DataDistribution import (
#    build_rhoY_gmm_with_corridor,
#    pushforward_Y_to_X_as_rho0,
#    rho0_mixture_mean_std,
#    set_rho0_standardization,
#    sample_x0_unlabeled,
#    sample_x0_labeled,
#    project_x_to_Y_via_U
#)


# -----------------------
# Global config (Colab)
# -----------------------
SEED   = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN    = torch.Generator(device=DEVICE).manual_seed(SEED)

# ----- Depth config (MLP blocks) -----
ENCODER_BLOCKS = 1      
DEC_BLOCKS     = 1      
d    = 56*56*3          

# ---------- Build data distribution in ambient X using NOT pushforward ----------
mu_Y_all, L_Y_all, w_Y_all, meta_slices, comp_to_label = build_rhoY_gmm_with_corridor(device=DEVICE)
mu_0, L_0_lowrank, w_0, U, b = pushforward_Y_to_X_as_rho0(
    mu_Y_all, L_Y_all, w_Y_all, d=d, device=DEVICE, seed=2024
)
mu_rho0, std_rho0 = rho0_mixture_mean_std(mu_0, L_0_lowrank, w_0)


set_rho0_standardization(mu_rho0, std_rho0)
def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - mu_rho0) / std_rho0

def un_zscore(x_std: torch.Tensor) -> torch.Tensor:
    return x_std * std_rho0 + mu_rho0

# -------------------
# Model components
# -------------------

# --------------------------
# SwiGLU activation (Shazeer 2020)
class SwiGLU(nn.Module):
    """
    Generalized SwiGLU that accepts 2*g*H input and returns H output by
    summing g gated pairs: sum_i SiLU(a_i) * b_i, optionally scaled.
    """
    def __init__(self, hidden: int, groups: int = 1, scale: bool = True):
        super().__init__()
        self.hidden = hidden
        self.groups = groups          # g = 1 -> 2H, g = 2 -> 4H, etc.
        self.scale = scale
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., 2*groups*hidden)
        chunks = x.split(self.hidden, dim=-1)       # list of length 2*groups
        assert len(chunks) == 2*self.groups, f"expected {2*self.groups} chunks, got {len(chunks)}"
        out = 0
        for i in range(self.groups):
            a, b = chunks[2*i], chunks[2*i+1]
            out = out + self.silu(a) * b
        if self.scale and self.groups > 1:
            out = out / math.sqrt(self.groups)      # keeps variance roughly stable
        return out

def build_mlp(in_dim: int, hidden: int, blocks: int,
              out_dim: int | None = None, act=nn.GELU, expand: int = 2):
    """
    If act is SwiGLU: block = Linear(last->H) -> Linear(H->expand*H) -> SwiGLU(H, groups=expand//2)
    Otherwise:        block = Linear(last->H) -> act()
    """
    layers = []
    last = in_dim
    for _ in range(blocks):
        if act is SwiGLU:
            assert expand % 2 == 0, "expand must be an even multiple (2, 4, 6, ...)"
            groups = expand // 2
            layers += [nn.Linear(last, hidden),
                       nn.Linear(hidden, expand * hidden),
                       SwiGLU(hidden, groups=groups)]
        else:
            layers += [nn.Linear(last, hidden), act()]
        last = hidden
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)

class Q_phi(nn.Module):  # q_φ(z|x)
    def __init__(self, d_x: int, d_z: int, h: int = 512):
        super().__init__()
        self.backbone = build_mlp(d_x, h, ENCODER_BLOCKS, out_dim=None, act=SwiGLU, expand=4)
        self.to_mu     = nn.Linear(h, d_z)
        self.to_logvar = nn.Linear(h, d_z)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.to_mu(h), self.to_logvar(h)

class T_theta(nn.Module):  # decoder mean T_θ(z) ≈ E[x|z]
    def __init__(self, d_z: int, d_x: int, h: int = 512):
        super().__init__()
        self.net = build_mlp(d_z, h, DEC_BLOCKS, out_dim=d_x, act=SwiGLU, expand=4)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class T_theta_cond(nn.Module):  # conditional decoder for CFG variant
    def __init__(self, d_z: int, d_x: int, n_classes: int = 3, cond_dim: int = 32, h: int = 512, null_id: int|None = None):
        super().__init__()
        self.null_id = (n_classes if null_id is None else null_id)  # reserve an id for ∅ token
        self.emb = nn.Embedding(n_classes + 1, cond_dim)            # +1 for ∅
        self.net = build_mlp(d_z + cond_dim, h, DEC_BLOCKS, out_dim=d_x, act=SwiGLU, expand=4)
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.emb(y)  # (B, cond_dim)
        return self.net(torch.cat([z, e], dim=-1))

class Q_psi(nn.Module):   # q_ψ(y|x) (simple linear head)
    def __init__(self, d_x: int, n_classes: int = 3, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x, h), nn.GELU(),
            nn.Linear(h, n_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------
# Utility: log densities
# ----------------------
def log_normal_diag(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # not in original math but had to add cause so much numerical instability
    logvar = logvar.clamp(min=-10.0, max=10.0)

    # log N(z; mu, diag(exp(logvar)))
    # replace ``1/logvar.exp()`` -> ``torch.exp(-logvar)`` to fix overflow loss problem. That's how bad VAE was!!!
    # return -0.5 * ( ((z - mu)**2 / logvar.exp()) + logvar + math.log(2*math.pi) ).sum(dim=-1)
    return -0.5 * ( ((z - mu)**2 * torch.exp(-logvar)) + logvar + math.log(2*math.pi) ).sum(dim=-1)

def log_p_lambda_mog(z: torch.Tensor, mu_k: torch.Tensor, logvar_k: torch.Tensor) -> torch.Tensor:
    """
    Mixture of Gaussians prior:
        p_λ(z) = (1/K) ∑_k N(z; μ_k, diag(exp(logσ²_k)))
    Returns log p_λ(z) for each z in the batch.
    Shapes:
        - z: (B, Z)
        - mu_k: (K, Z)
        - logvar_k: (K, Z)
    """
    logvar_k = logvar_k.clamp(min=-10.0, max=10.0)
    z_b = z.unsqueeze(1)                                # (B, 1, Z)
    mu  = mu_k.unsqueeze(0)                             # (1, K, Z)
    lv  = logvar_k.unsqueeze(0)                         # (1, K, Z)
    lp_mat = -0.5 * ( ((z_b - mu)**2 * torch.exp(-lv)) + lv + math.log(2*math.pi) ).sum(dim=-1)  # (B, K)
    return torch.logsumexp(lp_mat - math.log(mu_k.size(0)), dim=1)                               # (B,)


# ----------------------
# Exact FID in SI reference space 𝒴 (2D)
# ----------------------
@torch.no_grad()
def moments_rhoY_analytic(mu_Y_all: torch.Tensor, L_Y_all: torch.Tensor, w_Y_all: torch.Tensor):
    w = (w_Y_all / w_Y_all.sum()).to(mu_Y_all.dtype)
    mu = (w[:, None] * mu_Y_all).sum(dim=0)
    Sigma_intra = (w[:, None, None] * (L_Y_all @ L_Y_all.transpose(-1, -2))).sum(dim=0)
    mu_centered = mu_Y_all - mu
    Sigma_inter = (w[:, None, None] * (mu_centered[:, :, None] @ mu_centered[:, None, :])).sum(dim=0)
    Sigma = Sigma_intra + Sigma_inter
    return mu, Sigma

@torch.no_grad()
def moments_Y_from_generated(x_gen: torch.Tensor, U: torch.Tensor, b: torch.Tensor):
    y = (x_gen - b) @ U
    mu = y.mean(dim=0)
    yc = y - mu
    Sigma = (yc.T @ yc) / (max(1, y.shape[0] - 1))
    return mu, Sigma

@torch.no_grad()
def frechet_distance_2d(mu1: torch.Tensor, Sig1: torch.Tensor, mu2: torch.Tensor, Sig2: torch.Tensor, eps: float = 1e-9) -> float:
    I = torch.eye(2, device=mu1.device, dtype=mu1.dtype)
    Sig1 = Sig1 + eps * I
    Sig2 = Sig2 + eps * I
    e1, V1 = torch.linalg.eigh(Sig1)
    Sig1_half = (V1 * e1.clamp_min(0).sqrt()) @ V1.T
    M = Sig1_half @ Sig2 @ Sig1_half
    eM, VM = torch.linalg.eigh(M)
    sqrtM = (VM * eM.clamp_min(0).sqrt()) @ VM.T
    diff = (mu1 - mu2).unsqueeze(0)
    fd = (diff @ diff.T).item() + torch.trace(Sig1 + Sig2 - 2.0 * sqrtM).item()
    return float(fd)

@torch.no_grad()
def fid_in_Y_exact_real_generated(mu_Y_all, L_Y_all, w_Y_all, x_generated, U, b) -> float:
    muY_real, SigY_real = moments_rhoY_analytic(mu_Y_all, L_Y_all, w_Y_all)
    muY_gen,  SigY_gen  = moments_Y_from_generated(x_generated, U, b)
    return frechet_distance_2d(muY_real, SigY_real, muY_gen, SigY_gen)


# -----------------
# Training loops
# -----------------
def train_unsupervised_MogPrior(*, epochs=5, batch=384, Z_DIM=32, H=512, K_vamp=50, lr=3e-4, total_samples=25000, shuffle=True):
    q_phi = Q_phi(d, Z_DIM, h=H).to(DEVICE)
    Tdec  = T_theta(Z_DIM, d, h=H).to(DEVICE)
    mu_k  = torch.nn.Parameter(torch.randn(K_vamp, Z_DIM, device=DEVICE) * 0.1)  # MoG means
    logvar_k = torch.nn.Parameter(torch.zeros(K_vamp, Z_DIM, device=DEVICE))  # MoG log-variances
    opt   = torch.optim.AdamW(list(q_phi.parameters()) + list(Tdec.parameters()) + [mu_k, logvar_k], lr=lr)

    with torch.no_grad():
        X = zscore(sample_x0_unlabeled(total_samples, mu_0, L_0_lowrank, w_0, device=DEVICE))

    loader = DataLoader(TensorDataset(X), batch_size=batch, shuffle=shuffle, drop_last=False)

    loss_hist = []

    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    global_step = 0
    lr_history = []
    for ep in range(1, epochs+1):
        sum_loss, n_batches = 0.0, 0
        for (x_b,) in loader:
            # set scheduled LR
            lr_now = lr_warmup_linear_decay(global_step, total_steps)
            for g in opt.param_groups:
                g['lr'] = lr_now
            mu_z, logvar_z = q_phi(x_b.to(DEVICE))
            logvar_z = logvar_z.clamp(min=-10.0, max=10.0)
            z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)

            xhat = Tdec(z)
            recon = 0.5 * ((x_b.to(DEVICE) - xhat)**2).sum(dim=1)

            log_q = log_normal_diag(z, mu_z, logvar_z)
            log_p = log_p_lambda_mog(z, mu_k, logvar_k)
            loss = (recon + (log_q - log_p)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sum_loss += float(loss.detach().item()); n_batches += 1
            lr_history.append(lr_now)
            global_step += 1
        avg = sum_loss / max(1, n_batches)
        loss_hist.append(avg)
        print(f"[unsup] epoch {ep}/{epochs}  loss={avg:.3f}")
    return dict(q_phi=q_phi, Tdec=Tdec, mu_k=mu_k, logvar_k=logvar_k, loss_history=loss_hist, lr_history=lr_history)


def train_unsup_CFG_MogPrior(*, epochs=5, batch=384, Z_DIM=32, H=512, K_vamp=50, lr=3e-4, p_drop=0.2, total_samples=25000, shuffle=True):
    q_phi = Q_phi(d, Z_DIM, h=H).to(DEVICE)
    TdecC = T_theta_cond(Z_DIM, d, n_classes=3, cond_dim=32, h=H).to(DEVICE)
    mu_k  = torch.nn.Parameter(torch.randn(K_vamp, Z_DIM, device=DEVICE) * 0.1)
    logvar_k = torch.nn.Parameter(torch.zeros(K_vamp, Z_DIM, device=DEVICE))
    opt   = torch.optim.AdamW(list(q_phi.parameters()) + list(TdecC.parameters()) + [mu_k, logvar_k], lr=lr)

    with torch.no_grad():
        X_raw, Y = sample_x0_labeled(total_samples, mu_0, L_0_lowrank, w_0, comp_to_label, device=DEVICE)
        X = zscore(X_raw)

    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=shuffle, drop_last=False)

    loss_hist = []

    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    global_step = 0
    lr_history = []
    for ep in range(1, epochs+1):
        sum_loss, n_batches = 0.0, 0
        for x_b, y_b in loader:
            # set scheduled LR
            lr_now = lr_warmup_linear_decay(global_step, total_steps)
            for g in opt.param_groups:
                g['lr'] = lr_now
            mu_z, logvar_z = q_phi(x_b.to(DEVICE))
            logvar_z = logvar_z.clamp(min=-10.0, max=10.0)
            z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)

            # classifier-free conditioning: randomly replace labels with null token during training
            y_dec = y_b.clone().to(DEVICE)
            drop = (torch.rand_like(y_dec.float()) < p_drop)
            y_dec[drop] = TdecC.null_id

            xhat = TdecC(z, y_dec)
            recon = 0.5 * ((x_b.to(DEVICE) - xhat)**2).sum(dim=1)
            log_q = log_normal_diag(z, mu_z, logvar_z)
            log_p = log_p_lambda_mog(z, mu_k, logvar_k)
            loss  = (recon + (log_q - log_p)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sum_loss += float(loss.detach().item()); n_batches += 1
            lr_history.append(lr_now)
            global_step += 1
        avg = sum_loss / max(1, n_batches)
        loss_hist.append(avg)
        print(f"[unsup+CFG] epoch {ep}/{epochs}  loss={avg:.3f}")
    return dict(q_phi=q_phi, TdecC=TdecC, mu_k=mu_k, logvar_k=logvar_k, loss_history=loss_hist, lr_history=lr_history)



# -----------------
# Sampling helpers
# -----------------
@torch.no_grad()
def sample_prior_MoG(mu_k: torch.Tensor, logvar_k: torch.Tensor, n: int = 1000, generator=None) -> torch.Tensor:
    """
    Draw z ~ (1/K) ∑_k N(μ_k, diag(exp(logσ²_k))).
    mu_k, logvar_k: (K, Z). Returns z: (n, Z).
    """
    if generator is None:
        generator = GEN if 'GEN' in globals() else None
    K, Z = mu_k.shape
    idx = torch.randint(0, K, (n,), generator=generator, device=mu_k.device)
    mu = mu_k[idx]
    logvar = logvar_k[idx].clamp(min=-10.0, max=10.0)
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + eps * torch.exp(0.5 * logvar)

@torch.no_grad()
def generate_unsup(art, n=4000, Z_DIM=32):
    z = sample_prior_MoG(art['mu_k'], art['logvar_k'], n=n)
    xhat = art['Tdec'](z)
    return un_zscore(xhat)

@torch.no_grad()
def generate_unsup_cfg(art, n=4000, Z_DIM=32, target_class: int|None = None, guidance_w: float = 2.0):
    """
    Classifier-free guidance style: interpolate decoder outputs between null and class condition.
    If target_class is None, we draw classes uniformly for diversity.
    """
    z = sample_prior_MoG(art['mu_k'], art['logvar_k'], n=n)
    TdecC: T_theta_cond = art["TdecC"]
    if target_class is None:
        y = torch.randint(0, TdecC.emb.num_embeddings-1, (n,), device=DEVICE, generator=GEN)
    else:
        y = torch.full((n,), int(target_class), device=DEVICE, dtype=torch.long)
    y_null = torch.full_like(y, TdecC.null_id)

    x_null = TdecC(z, y_null)
    x_cond = TdecC(z, y)
    x_cfg  = x_null + guidance_w * (x_cond - x_null)
    return un_zscore(x_cfg), y

# ---------------
# Main (Colab)
# ---------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    total_samples = 1000
    K_vamp=1000

    colors = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}
    names  = {0: "Blue cluster", 1: "Red cluster", 2: "Green cluster"}

    EPOCHS = 300  

    art_unsup = train_unsupervised_MogPrior(epochs=EPOCHS, batch=384, Z_DIM=32, H=512, K_vamp=K_vamp, total_samples=total_samples, lr=LR_MAX)

    art_cfg   = train_unsup_CFG_MogPrior(epochs=EPOCHS, batch=384, Z_DIM=32, H=512, K_vamp=K_vamp, lr=LR_MAX, p_drop=0.2, total_samples=total_samples)


    # plot training loss
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(art_unsup["loss_history"], label="Unsupervised (ELBO)", linewidth=1.5)
    plt.plot(art_cfg["loss_history"],   label="Unsupervised + CFG (ELBO)", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Training loss")
    plt.title("Training loss per epoch")
    plt.grid(True, ls="--", alpha=0.3); plt.legend(loc="best")
    plt.show()


    # Plot approximated data distribution
    ### (i) unsupervised
    X_gen_u = generate_unsup(art_unsup, n=4000)
    Y_gen_u = project_x_to_Y_via_U(X_gen_u, U, b).detach().cpu()
    plt.figure(figsize=(6.5,6.5))
    plt.scatter(Y_gen_u[:,0], Y_gen_u[:,1], s=2, alpha=0.55, c="tab:purple", label="VAE+VampPrior (gen)")
    plt.gca().set_aspect("equal"); plt.grid(True, ls="--", alpha=0.3)
    plt.title("Unsupervised: approximated data distribution")
    plt.legend(loc="upper right"); plt.show()

    ### (ii) unsupervised + CFG
    X_gen_c, y_cls = generate_unsup_cfg(art_cfg, n=4000, target_class=None, guidance_w=2.0)
    Y_gen_c = project_x_to_Y_via_U(X_gen_c, U, b).detach().cpu()
    plt.figure(figsize=(6.5,6.5))
    for cls in [0,1,2]:
        m = (y_cls.cpu() == cls)
        if m.any():
            plt.scatter(Y_gen_c[m,0], Y_gen_c[m,1], s=3, alpha=0.75, c=colors[cls], marker="x", label=f"CFG gen (class {names[cls]})")
    plt.gca().set_aspect("equal"); plt.grid(True, ls="--", alpha=0.3)
    plt.title("Unsupervised + CFG: approximated data distribution")
    plt.legend(loc="upper right", fontsize=9); plt.show()

    # Compute FID evaluation: Use exact FID in 𝒴 true data dsitribution vs approximated data distribution
    fidY_unsup = fid_in_Y_exact_real_generated(mu_Y_all, L_Y_all, w_Y_all, X_gen_u, U, b)
    print(f"FID-Y (analytic true in 𝒴 vs. generated unsup): {fidY_unsup:.6f}")
    fidY_cfg = fid_in_Y_exact_real_generated(mu_Y_all, L_Y_all, w_Y_all, X_gen_c, U, b)
    print(f"FID-Y (analytic true in 𝒴 vs. generated CFG):   {fidY_cfg:.6f}")