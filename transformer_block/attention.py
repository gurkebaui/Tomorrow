import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, 1, 1, seq_len) or broadcastable

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights for visualization of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # Reshape for multi-head attention: (B, S, H, D//H) -> (B, H, S, D//H)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (B, H, S, S)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, D//H)

        # Reshape back: (B, H, S, D//H) -> (B, S, H, D//H) -> (B, S, D)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output, attn_weights


class CausalAttention(nn.Module):
    """Causal (masked) multi-head attention for autoregressive generation."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape

        # Create causal mask (lower triangular) to prevent attending to future tokens
        mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, S, S)

        return self.attention(x, mask=mask)
