"""FAENet: single evidential feed-forward network (plus a softmax ablation head, variant F)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.1,
                 head: str = "evidential"):
        super().__init__()
        assert head in ("evidential", "softmax")
        self.head = head
        layers, d = [], input_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 2)  # class 0 = stayed, class 1 = left

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.body(x)
        z = self.out(h)
        if self.head == "evidential":
            evidence = F.softplus(z)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S
            uncertainty = 2.0 / S.squeeze(1)           # u = K / S, K = 2
            logodds = torch.log(alpha[:, 1]) - torch.log(alpha[:, 0])
            return {"logits": z, "evidence": evidence, "alpha": alpha, "probabilities": probs,
                    "uncertainty": uncertainty, "logodds": logodds, "embedding": h}
        probs = F.softmax(z, dim=1)
        # normalised binary entropy in [0, 1] as the softmax model's review score
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(1) / torch.log(torch.tensor(2.0))
        return {"logits": z, "probabilities": probs, "uncertainty": ent, "logodds": z[:, 1] - z[:, 0], "embedding": h}
