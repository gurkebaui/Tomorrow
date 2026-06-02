import torch
import torch.nn as nn

from .attention import CausalAttention
from .mlp import FeedForward
from .norm import LayerNorm


class TransformerBlock(nn.Module):
    """A single transformer block with causal attention, layer norm, and MLP.

    Architecture: Pre-LN (LayerNorm before each sub-layer)
        x -> Attention -> Add & Norm -> MLP -> Add & Norm -> Output
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = CausalAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.attn_norm = LayerNorm(d_model=d_model)
        self.mlp = FeedForward(d_model=d_model, d_ff=d_model * 4, dropout=dropout)
        self.mlp_norm = LayerNorm(d_model=d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Transformer block output of the same shape as input
            attn_weights: Attention weights for visualization
        """
        # Pre-LN attention sub-layer with residual connection
        attn_output, attn_weights = self.attention(x)
        x = x + self.dropout1(attn_output)  # Residual
        x = self.attn_norm(x)

        # Pre-LN MLP sub-layer with residual connection
        mlp_output = self.mlp(x)
        x = x + self.dropout2(mlp_output)  # Residual
        x = self.mlp_norm(x)

        return x, attn_weights
