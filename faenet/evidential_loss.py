"""Evidential classification loss (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

The data-fit term and the KL regulariser are returned separately and combined exactly once
in the training loop:  L = L_fit^weighted + lambda_r(t) * L_KL + lambda_f * L_fair.
"""
from __future__ import annotations

import torch


def edl_mse_fit(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Per-sample expected-MSE evidential loss (Sensoy Eq. 5)."""
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = (y_onehot - p) ** 2
    var = p * (1 - p) / (S + 1)
    return (err + var).sum(dim=1)


def edl_log_fit(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Per-sample Type-II maximum-likelihood evidential loss (Sensoy Eq. 3), for sensitivity analysis."""
    S = alpha.sum(dim=1, keepdim=True)
    return (y_onehot * (torch.log(S) - torch.log(alpha))).sum(dim=1)


def kl_to_uniform_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """Per-sample KL[ Dir(alpha) || Dir(1) ]."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    term1 = torch.lgamma(S.squeeze(1)) - torch.lgamma(alpha).sum(dim=1) - torch.lgamma(
        torch.tensor(float(K), device=alpha.device))
    term2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return term1 + term2


def adjusted_alpha(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """alpha_tilde = y + (1 - y) * alpha: removes the evidence of the true class."""
    return y_onehot + (1 - y_onehot) * alpha


def kl_coefficient(epoch: int, anneal_epochs: int, lambda_r_max: float) -> float:
    if anneal_epochs <= 0:
        return lambda_r_max
    return min(1.0, epoch / anneal_epochs) * lambda_r_max


def evidential_terms(alpha: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor | None,
                     fit: str = "mse") -> dict[str, torch.Tensor]:
    """Returns weighted mean fit loss and (unweighted) mean KL. KL is NOT added here."""
    y_oh = torch.nn.functional.one_hot(y.long(), num_classes=2).float()
    per = edl_mse_fit(alpha, y_oh) if fit == "mse" else edl_log_fit(alpha, y_oh)
    if class_weights is not None:
        per = per * class_weights[y.long()]
    kl = kl_to_uniform_dirichlet(adjusted_alpha(alpha, y_oh))
    return {"fit": per.mean(), "kl": kl.mean()}


def softmax_terms(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor | None):
    """Variant F: class-weighted cross-entropy (mean of per-sample weighted CE)."""
    per = torch.nn.functional.cross_entropy(logits, y.long(), reduction="none")
    if class_weights is not None:
        per = per * class_weights[y.long()]
    return {"fit": per.mean(), "kl": torch.zeros((), device=logits.device)}
