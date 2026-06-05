import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class LongConvolution(nn.Module):
    """
    Long Convolution block for global pattern capture.
    Uses time-domain depthwise Conv1d which is 100% ONNX exportable.
    """
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Learnable filter weights in time domain
        # Shape: [d_model, 1, seq_len] for depthwise convolution
        self.filter_weights = nn.Parameter(torch.randn(d_model, 1, seq_len) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
        Returns:
            out: Convolved tensor of shape [B, L, D]
        """
        B, L, D = x.shape
        # Transpose to [B, D, L] for depthwise conv along sequence dimension L
        x_trans = x.transpose(1, 2)
        
        # Non-causal padding of L-1 on both sides
        x_padded = F.pad(x_trans, (L - 1, L - 1), mode="constant", value=0.0)
        
        # Depthwise 1D convolution: groups = D
        # self.filter_weights shape is [D, 1, L]
        y = F.conv1d(x_padded, self.filter_weights, groups=self.d_model)
        
        # Extract the center L elements (matches non-causal global context)
        out = y[..., L - 1 : 2 * L - 1] + self.bias.view(1, D, 1)
        
        # Transpose back to [B, L, D]
        return out.transpose(1, 2)


class DeltaNetBlock(nn.Module):
    """
    DeltaNet block implementing linear RNN state updates using the delta rule.
    Includes a fast native PyTorch recurrent implementation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Projections for Queries, Keys, Values, and Beta gate
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, 1, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
        Returns:
            out: RNN state sequence of shape [B, L, D]
        """
        B, L, D = x.shape
        
        Q = self.q_proj(x) # [B, L, D]
        # Normalize Keys to prevent gradient/value explosion
        K = F.normalize(self.k_proj(x), p=2, dim=-1) # [B, L, D]
        V = self.v_proj(x) # [B, L, D]
        Beta = torch.sigmoid(self.beta_proj(x)) # [B, L, 1]
        
        # Recurrent state update W of shape [B, D, D] initialized to zero
        W = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outputs = []
        
        # Recurrence loop
        for t in range(L):
            q_t = Q[:, t, :].unsqueeze(-1) # [B, D, 1]
            k_t = K[:, t, :].unsqueeze(-1) # [B, D, 1]
            v_t = V[:, t, :].unsqueeze(-1) # [B, D, 1]
            beta_t = Beta[:, t, :].unsqueeze(-1) # [B, 1, 1]
            
            # Reconstructed value: W_{t-1} @ k_t
            v_pred = torch.matmul(W, k_t) # [B, D, 1]
            
            # Delta rule update: W_t = W_{t-1} + beta_t * (v_t - v_pred) @ k_t^T
            delta = beta_t * torch.matmul((v_t - v_pred), k_t.transpose(-1, -2)) # [B, D, D]
            W = W + delta
            
            # Retrieve output: y_t = W_t @ q_t
            y_t = torch.matmul(W, q_t) # [B, D, 1]
            outputs.append(y_t.squeeze(-1))
            
        out_rnn = torch.stack(outputs, dim=1) # [B, L, D]
        return self.out_proj(out_rnn)


class GatedMLP(nn.Module):
    """
    Gated MLP block for non-linear channel-mixing feature interactions.
    """
    def __init__(self, d_model: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        self.fc1 = nn.Linear(d_model, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
        """
        hidden = self.fc1(x)
        # Gate and value branches
        gate, val = hidden.chunk(2, dim=-1)
        # Apply SiLU gating
        x_gated = F.silu(gate) * val
        out = self.fc2(x_gated)
        return self.dropout(out)


class GatedRouter(nn.Module):
    """
    Learned Grouting/Routing block to dynamically blend the outputs
    of Long Convolution, DeltaNet, and Gated MLP modules.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.router = nn.Linear(d_model, 3)
        
    def forward(self, x: torch.Tensor, out_conv: torch.Tensor, out_rnn: torch.Tensor, out_mlp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Original layer input [B, L, D] (used to compute routing weights)
            out_conv: Output of Long Conv [B, L, D]
            out_rnn: Output of DeltaNet [B, L, D]
            out_mlp: Output of Gated MLP [B, L, D]
        Returns:
            routed: Blended output [B, L, D]
        """
        # Compute dynamic routing weights based on average pooling of the input
        x_summary = x.mean(dim=1)  # [B, D]
        logits = self.router(x_summary)  # [B, 3]
        weights = F.softmax(logits, dim=-1)  # [B, 3]
        
        w_conv = weights[:, 0].view(-1, 1, 1)
        w_rnn = weights[:, 1].view(-1, 1, 1)
        w_mlp = weights[:, 2].view(-1, 1, 1)
        
        return w_conv * out_conv + w_rnn * out_rnn + w_mlp * out_mlp


class SequenceMixingBlock(nn.Module):
    """
    Sequence mixing block integrating Conv, RNN, and MLP sub-layers with dynamic routing.
    """
    def __init__(self, seq_len: int, d_model: int, expansion_factor: int = 2, dropout: float = 0.1, use_router: bool = True):
        super().__init__()
        self.use_router = use_router
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
        self.conv = LongConvolution(seq_len, d_model)
        self.rnn = DeltaNetBlock(d_model)
        self.mlp = GatedMLP(d_model, expansion_factor, dropout)
        
        if use_router:
            self.router = GatedRouter(d_model)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape [B, L, D]
        """
        # Run sub-modules with residual connections and norms
        out_conv = self.conv(self.norm1(x))
        out_rnn = self.rnn(self.norm2(x))
        out_mlp = self.mlp(self.norm3(x))
        
        if self.use_router:
            routed = self.router(x, out_conv, out_rnn, out_mlp)
        else:
            # Equal weighting baseline
            routed = (out_conv + out_rnn + out_mlp) / 3.0
            
        return x + routed
