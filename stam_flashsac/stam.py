import torch
import torch.nn as nn


class STAMContextModule(nn.Module):
    """Short-term action-observation memory fusion for actor and critic features."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        feature_dim: int,
        context_len: int,
        n_models: int | None = None,
        encoder: str = "lstm",
        fusion: str = "gate",
    ):
        super().__init__()
        self.n_models = n_models
        self.encoder_type = encoder
        self.fusion = fusion
        if encoder == "lstm":
            self.encoder = nn.LSTM(context_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.encoder = nn.GRU(context_dim, hidden_dim, batch_first=True)
        elif encoder == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(context_len * context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            raise ValueError(f"Unsupported STAM encoder: {encoder}")

        if fusion == "gate":
            out_dim = feature_dim if n_models is None else n_models * feature_dim
            self.proj = nn.Linear(hidden_dim, out_dim)
        elif fusion == "concat":
            self.proj = nn.Linear(hidden_dim + feature_dim, feature_dim)
        else:
            raise ValueError(f"Unsupported STAM fusion: {fusion}")

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder(context)
            return h_n[-1]
        if self.encoder_type == "gru":
            _, h_n = self.encoder(context)
            return h_n[-1]
        return self.encoder(context.flatten(start_dim=1))

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if context is None:
            return features
        summary = self.encode_context(context)
        if self.fusion == "gate":
            gate = torch.sigmoid(self.proj(summary))
            if self.n_models is None:
                return features * gate
            gate = gate.view(context.shape[0], self.n_models, -1).permute(1, 0, 2)
            return features * gate

        if self.n_models is None:
            fused = torch.cat([features, summary], dim=-1)
            return self.proj(fused)
        expanded_summary = summary.unsqueeze(0).expand(self.n_models, -1, -1)
        fused = torch.cat([features, expanded_summary], dim=-1)
        return self.proj(fused)


STAMContextGate = STAMContextModule
