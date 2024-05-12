import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import pdb

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.model_dim % config.n_head == 0
        self.QKV = nn.Linear(config.model_dim, 3 * config.model_dim, bias=config.bias)
        self.proj = nn.Linear(config.model_dim, config.model_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
    
    def forward(self, x_BTD):
        B, T, D = x_BTD.shape
        H = D // self.n_head
        # compute queries, keys, and values per head
        q_BTD, k_BTD, v_BTD = self.QKV(x_BTD).chunk(3, dim=-1)
        q_BNTH = q_BTD.contiguous().view(B, self.n_head, T, H)
        k_BNTH = k_BTD.contiguous().view(B, self.n_head, T, H)
        v_BNTH = v_BTD.contiguous().view(B, self.n_head, T, H)
        # compute attention score
        a_BNTT = torch.einsum("bnih,bnjh->bnij", q_BNTH, k_BNTH)
        a_BNTT = torch.softmax(a_BNTT / math.sqrt(D), dim=3)
        a_BNTT = self.attn_dropout(a_BNTT)
        # compute output and recombine heads
        o_BNTH = torch.einsum("bntt,bnth->bnth", a_BNTT, v_BNTH)
        o_BTD = o_BNTH.transpose(1, 2).contiguous().view(B, T, D)
        out_BTD = self.resid_dropout(self.proj(o_BTD))
        return out_BTD


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, 4 * config.model_dim),
            nn.GELU(),
            nn.Linear(4 * config.model_dim, config.model_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x_BTD):
        out_BTD = self.ffn(x_BTD)
        return out_BTD


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.attn = SelfAttention(config)
        self.ffn = FFN(config)
    
    def forward(self, x_BTD):
        x_BTD = x_BTD + self.attn(self.ln1(x_BTD))
        out_BTD = x_BTD + self.ffn(self.ln2(x_BTD))
        return out_BTD


@dataclass
class ViTConfig:
    n_channels: int = 3
    patch_size: int = 16
    model_dim: int = 64
    n_head: int = 16
    n_layers: int = 6
    n_classes: int = 10
    seq_len: int = 4
    dropout: float = 0.1
    bias: bool = True


class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        token_dim = config.n_channels * config.patch_size ** 2
        self.token_embed = nn.Linear(token_dim, config.model_dim, bias=False)
        self.pos_embed = nn.Embedding(config.seq_len + 1, config.model_dim)
        self.cls_embed_11D = nn.Parameter(0.1 * torch.ones(1, 1, config.model_dim))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.n_classes)
        )
    
    def forward(self, x_BNI, y_B=None):
        # add classification token and embed image tokens/patches
        B, N, _ = x_BNI.shape
        T = N + 1
        x_BND = self.token_embed(x_BNI)
        D = x_BND.shape[2]
        x_BTD = torch.cat([self.cls_embed_11D.expand(B, 1, D), x_BND], dim=1)
        # add positional embedding
        x_BTD = x_BTD + self.pos_embed(torch.arange(T, device=x_BTD.device))
        # apply blocks
        for block in self.blocks:
            x_BTD = block(x_BTD)
        # apply head on first (classification) position in the sequence
        logits_BC = self.head(x_BTD[:, 0, :])
        loss = None
        if y_B is not None:
            loss = F.cross_entropy(logits_BC, y_B)
        return logits_BC, loss
