from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from RMSNorm import RMSNorm
from rotary_embedding import apply_rotary_emb


@dataclass
class ModelConfig:
    dim: int = 128
    n_heads: int = 12
    n_layers: int = 28


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.wg = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.wg(x) * self.w1(x)))


class MultiHeadDifferAttention(nn.Module):
    def __init__(self, args: ModelConfig, depth):
        super().__init__()
        self.dim = ModelConfig.dim
        self.to_q = nn.Linear(self.dim, self.dim)
        self.to_k = nn.Linear(self.dim, self.dim)
        self.to_v = nn.Linear(self.dim, self.dim)
        self.head_dim = self.dim // args.n_heads // 2
        self.num_heads = args.n_heads
        self.scale = self.head_dim**-0.5

        self.lamda_init = 0.8 - 0.6 * torch.exp(-0.3 * depth)
        self.lamda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lamda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lamda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lamda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

    def forward(self, x: torch.tensor, freqs_cis: torch.tensor):
        bsz, seqlen, _ = x.size()
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(bsz, seqlen, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, 2 * self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, 2 * self.num_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        q = q * self.scale

        attn = torch.matmul(q, k.transpose(2, 3))
        attn_mask = torch.triu(
            torch.zeros([seqlen, seqlen]).float().fill_(float("inf")).type_as(attn)
        )
        attn = attn.nan_to_num(attn)
        attn += attn_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        lambda_ = (
            torch.exp(self.lamda_q1 * self.lamda_k1)
            - torch.exp * (self.lamda_q2 * self.lamda_k2)
            + self.lamda_init
        )
        attn = attn.view(bsz, seqlen, self.num_heads, 2, self.head_dim)
