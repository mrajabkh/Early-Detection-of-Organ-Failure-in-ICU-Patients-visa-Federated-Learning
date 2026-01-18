# gru_risk.py
# Simple supervised GRU for early detection on windowed time series

from __future__ import annotations

import torch
import torch.nn as nn


class GRURisk(nn.Module):
    """
    GRU-based risk predictor.
    Input:  (B, T, D)
    Output: logits (B, T)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, D)
        lengths: (B,) true sequence lengths (optional)

        returns:
          logits: (B, T)
        """
        if lengths is not None:
            # Pack sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
            )
        else:
            out, _ = self.gru(x)

        out = self.dropout(out)
        logits = self.head(out).squeeze(-1)  # (B, T)
        return logits
