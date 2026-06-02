import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network (MLP)."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of the same shape as input
        """
        return self.linear2(self.gelu(self.dropout(self.linear1(x))))
