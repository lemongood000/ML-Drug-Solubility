from __future__ import annotations

import torch
from torch import nn


class TwoTowerRegressor(nn.Module):
    def __init__(
        self,
        drug_dim: int,
        solvent_dim: int,
        temperature_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        encoder_dropout: float = 0.0,
        interaction_dim: int = 0,
    ) -> None:
        super().__init__()
        self.interaction_dim = interaction_dim
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.solvent_encoder = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temperature_encoder = nn.Sequential(
            nn.Linear(temperature_dim, 16),
            nn.GELU(),
        )
        interaction_features = 0
        if interaction_dim > 0:
            self.drug_interaction = nn.Linear(hidden_dim, interaction_dim)
            self.solvent_interaction = nn.Linear(hidden_dim, interaction_dim)
            interaction_features = interaction_dim
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + 16 + interaction_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        drug_features: torch.Tensor,
        solvent_features: torch.Tensor,
        temperature_features: torch.Tensor,
    ) -> torch.Tensor:
        drug_embedding = self.drug_encoder(drug_features)
        solvent_embedding = self.solvent_encoder(solvent_features)
        temperature_embedding = self.temperature_encoder(temperature_features)
        interaction = []
        if self.interaction_dim > 0:
            interaction = [
                self.drug_interaction(drug_embedding)
                * self.solvent_interaction(solvent_embedding)
            ]
        fused = torch.cat(
            [drug_embedding, solvent_embedding, temperature_embedding, *interaction],
            dim=1,
        )
        return self.regressor(fused).squeeze(1)
