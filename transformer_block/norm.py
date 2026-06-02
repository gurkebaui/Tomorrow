import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift parameters."""

    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized output of the same shape as input
        """
        return self.norm(x)
