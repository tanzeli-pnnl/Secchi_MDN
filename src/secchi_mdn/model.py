"""PyTorch Mixture Density Network for 1-D Secchi depth."""

from __future__ import annotations

from dataclasses import dataclass


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for MDN training. Install it first, for example "
            "with `pip install torch` or by using `pip install -r requirements.txt`."
        ) from exc
    return torch, nn


@dataclass
class MDNOutputs:
    pi_logits: object
    mu: object
    log_sigma: object


class SecchiMDNFactory:
    """Factory so the module can be imported without torch installed."""

    @staticmethod
    def create(input_dim: int, n_mix: int, hidden_dims: list[int]):
        torch, nn = require_torch()

        class SecchiMDN(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                current_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    current_dim = hidden_dim
                self.backbone = nn.Sequential(*layers)
                self.pi_head = nn.Linear(current_dim, n_mix)
                self.mu_head = nn.Linear(current_dim, n_mix)
                self.log_sigma_head = nn.Linear(current_dim, n_mix)

            def forward(self, x):
                hidden = self.backbone(x)
                return MDNOutputs(
                    pi_logits=self.pi_head(hidden),
                    mu=self.mu_head(hidden),
                    log_sigma=self.log_sigma_head(hidden).clamp(min=-7.0, max=5.0),
                )

        return torch, SecchiMDN()


def mdn_nll_loss(outputs: MDNOutputs, target, epsilon: float = 1.0e-6):
    """Negative log-likelihood for a 1-D Gaussian mixture."""
    torch, _ = require_torch()
    target = target.unsqueeze(-1)
    sigma = torch.exp(outputs.log_sigma) + epsilon
    log_pi = torch.log_softmax(outputs.pi_logits, dim=-1)
    normal = torch.distributions.Normal(outputs.mu, sigma)
    component_log_prob = normal.log_prob(target) + log_pi
    return -torch.logsumexp(component_log_prob, dim=-1).mean()


def mdn_predict(outputs: MDNOutputs, mode: str = "top"):
    """Return mixture predictions in transformed target space."""
    torch, _ = require_torch()
    pi = torch.softmax(outputs.pi_logits, dim=-1)
    if mode == "mean":
        return (pi * outputs.mu).sum(dim=-1)
    if mode != "top":
        raise ValueError(f"Unknown prediction mode '{mode}'. Expected 'top' or 'mean'.")
    top_index = torch.argmax(pi, dim=-1, keepdim=True)
    return torch.gather(outputs.mu, dim=-1, index=top_index).squeeze(-1)
