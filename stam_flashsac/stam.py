import torch
import torch.nn as nn


class STAMContextGate(nn.Module):
    """Short-term action-observation memory gate for actor and critic features."""

    def __init__(self, context_dim: int, hidden_dim: int, feature_dim: int, n_models: int | None = None):
        super().__init__()
        self.n_models = n_models
        self.lstm = nn.LSTM(context_dim, hidden_dim, batch_first=True)
        if n_models is None:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Linear(hidden_dim, n_models * feature_dim)

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if context is None:
            return features
        _, (h_n, _) = self.lstm(context)
        gate = torch.sigmoid(self.proj(h_n[-1]))
        if self.n_models is None:
            return features * gate
        gate = gate.view(context.shape[0], self.n_models, -1).permute(1, 0, 2)
        return features * gate
