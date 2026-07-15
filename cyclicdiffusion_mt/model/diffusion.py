"""DDPM diffusion processes: wrapped normal (continuous) + masked discrete."""
import torch, torch.nn as nn, torch.nn.functional as F, math

def cosine_schedule(T=500, s=0.008):
    """Cosine noise schedule. Returns alpha_bar_t for t=0..T-1."""
    steps = torch.arange(T+1, dtype=torch.float32)
    alpha_bar = torch.cos((steps/T + s) / (1+s) * math.pi/2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar[:T]  # shape (T,)

class WrappedNormalDiffusion(nn.Module):
    """Wrapped normal diffusion for torsion angles on [-pi, pi)."""
    def __init__(self, T=500):
        super().__init__()
        self.T = T
        alpha_bar = cosine_schedule(T)
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-8)
        beta = torch.clamp(beta, max=0.999)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta', beta)

    def q_sample(self, tau_0, t):
        """Forward noising: tau_0 (B,L,7) + t (B,) -> tau_t, noise.

        For wrapped normal diffusion on angular variables, the forward
        process adds noise centered at tau_0 — we do NOT scale tau_0 by
        sqrt(alpha_bar), which would bias angles toward zero.
        """
        a_bar = self.alpha_bar[t]  # (B,)
        noise = torch.randn_like(tau_0)
        tau_t = tau_0 + noise * (1 - a_bar).view(-1, 1, 1).sqrt()
        tau_t = torch.atan2(torch.sin(tau_t), torch.cos(tau_t))
        return tau_t, noise

    def loss(self, tau_0, tau_pred):
        """Wrapped L2 loss: min_k ||tau_0 - tau_pred + 2*pi*k||^2."""
        diff = tau_0 - tau_pred
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return (diff ** 2).mean()

class MaskedDiscreteDiffusion(nn.Module):
    """Masked discrete diffusion for amino acid types. D3PM-style."""
    def __init__(self, T=500, num_classes=26, mask_idx=25):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.mask_idx = mask_idx
        alpha_bar = cosine_schedule(T)
        self.register_buffer('alpha_bar', alpha_bar)

    def q_sample(self, a_0, t):
        """Forward: a_0 (B,L) + t (B,) -> a_t (B,L) with mask corruption."""
        a_bar = self.alpha_bar[t]  # (B,)
        # With prob 1-a_bar, replace with MASK
        mask_prob = (1 - a_bar).view(-1,1)
        rand = torch.rand_like(a_0.float())
        a_t = torch.where(rand < mask_prob,
                          torch.full_like(a_0, self.mask_idx),
                          a_0)
        return a_t

    def loss(self, a_0, logits):
        """Cross-entropy loss for AA type prediction. a_0:(B,L) logits:(B,L,26)."""
        return F.cross_entropy(logits.view(-1, self.num_classes), a_0.view(-1),
                               ignore_index=self.mask_idx)

    @torch.no_grad()
    def p_sample(self, logits, a_t, t):
        """Sample a_{t-1} from predicted a_0 logits. logits:(B,L,26) a_t:(B,L) t:(B,)."""
        probs = F.softmax(logits, dim=-1)
        a_pred = torch.multinomial(probs.view(-1, self.num_classes), 1).view_as(a_t)
        # For non-masked positions, keep original (only unmask masked ones)
        a_out = torch.where(a_t == self.mask_idx, a_pred, a_t)
        return a_out
