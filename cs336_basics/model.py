import math
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

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
    def __init__(self, d_model, eps: float=1e-5, device=None, dtype=None):
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
        k = torch.arange(1, self.d_k/2 + 1, dtype=torch.float32)
        theta_k = 1 / torch.pow(self.theta, torch.div(2*k - 2, self.d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)  # (max_seq_len, 1)
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

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    temp_features = in_features - in_features.max(dim=dim, keepdim=True).values
    return temp_features.exp() / temp_features.exp().sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    attention = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1]) # [... queries, keys]
    masked_attention = attention.masked_fill(~mask, float('-inf'))
    result = softmax(masked_attention, dim=-1) @ V
    return result

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=None, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.w_q = nn.Parameter(torch.randn(num_heads * self.d_k, self.d_model))
        self.w_k = nn.Parameter(torch.randn(num_heads * self.d_k, self.d_model))
        self.w_v = nn.Parameter(torch.randn(num_heads * self.d_v, self.d_model))
        self.w_o = nn.Parameter(torch.randn(d_model, num_heads * self.d_v))
        if max_seq_length and theta:
            self.rope = RoPE(theta, self.d_k, max_seq_length)
    
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        Q = in_features @ self.w_q.T # [batch, sequence_length, num_heads * d_k]
        K = in_features @ self.w_k.T
        V = in_features @ self.w_v.T
        
        B, S, _ = in_features.shape
        Q = Q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, S, d_k]
        K = K.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.d_v).transpose(1, 2)
        if self.rope:
            token_positions = torch.arange(S)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        mask = torch.tril(torch.ones(S, S)).bool()
        mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, S, S]
        att = scaled_dot_product_attention(Q, K, V, mask) #[B, H, S, d_v]
        att = att.transpose(1, 2).reshape(B, S, self.num_heads * self.d_v)
        output = att @ self.w_o.T
        return output
        