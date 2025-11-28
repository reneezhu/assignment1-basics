import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.empty(out_features, in_features)
        theta = 2/(in_features + out_features)
        nn.init.trunc_normal_(w)
        self.weights = nn.Parameter(w)
    
    def forward(self, x: torch.Tensor):
        return x @ self.weights.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(w, a=-3, b=3)
        self.weights = nn.Parameter(w)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor): 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(((x**2).sum(dim=-1) + self.eps)/self.d_model)
        result = x * self.g / rms.unsqueeze(-1)
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff))
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model))
    
    def silu(self, x:torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
    
class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute the sin & cos for all position
        k = torch.arange(1, self.d_k/2 + 1)
        theta_k = 1 / self.theta**((2*k - 2)/self.d_k)
        positions = torch.arange(max_seq_len).unsqueeze(-1)  # (max_seq_len, 1)
        angles = positions * theta_k  # (max_seq_len, d_k/2)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        sin_pos = self.sin[token_positions]
        cos_pos = self.cos[token_positions]
        
        # Consider a pair [x_even, x_odd], R = [[cos, -sin],[sin, cos]]
        # After the transformation, the pari becomes
        # [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos
        
        # Stack the even and odd parts back together along the last dimension and reshape
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(*x.shape[:-1], -1)
        
        return x_rot
