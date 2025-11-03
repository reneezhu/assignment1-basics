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
    
    def rotation_matrix(self, theta_k):
        result = torch.zeros(self.d_k, self.d_k)
        for k in range(1, theta_k.shape[0] + 1):
            result[2*k-2][2*k-2] = torch.cos(theta_k[k-1])
            result[2*k-1][2*k-2] = -torch.sin(theta_k[k-1])
            result[2*k-2][2*k-1] = torch.sin(theta_k[k-1])
            result[2*k-1][2*k-1] = torch.cos(theta_k[k-1])
        return result

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        k = torch.arange(1, self.d_k/2 + 1)
        theta_k = 1 / self.theta**((2*k - 2)/self.d_k)
        r = [] # (max_seq_len, d_k, d_k)
        for i in torch.arange(self.max_seq_len):
            R_i = self.rotation_matrix(theta_k * i)
            r.append(R_i)
        result = torch.zeros(x.shape)
        for b, sequence in enumerate(x):
            for x in range(sequence.shape[0]):
                result[b][x] = sequence[x] @ r[token_positions[x]]
        return result
