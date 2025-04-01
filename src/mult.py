import math
import torch
import torch.nn as nn
from typing import List

class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 2)
        x = self.conv1d(x)
        return x.permute(1, 0, 2).squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, feature_dim: int):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        pe = torch.zeros(seq_len, feature_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * -math.log(10000.0) / feature_dim
        )

        angles = position * div_term
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == torch.Size([self.seq_len, self.feature_dim]), "Tensor Shape is incorrect"
        return x + self.pe

class CrossModalTransformerLayer(nn.Module):
    def __init__(self, primary_in_shape, secondary_in_shape, dims: int):
        super(CrossModalTransformerLayer, self).__init__()
        self.dims = dims
        self.primary_ln = nn.LayerNorm(primary_in_shape)
        self.secondary_ln = nn.LayerNorm(secondary_in_shape)
        self.primary_proj = nn.Linear(in_features=primary_in_shape[1], out_features=dims)
        self.secondary_proj = nn.Linear(in_features=secondary_in_shape[1], out_features=dims*2)
        self.softmax = nn.Softmax()

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        atten = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dims)
        atten = self.softmax(atten)
        return atten @ V

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        primary = self.primary_ln(primary)
        secondary = self.secondary_ln(secondary)
        Q = self.primary_proj(primary)
        KV = self.secondary_proj(secondary)
        K, V = torch.split(KV, KV.size(1) // 2, dim=1)

        atten = self.attention(Q, K, V)

        return atten + Q

class CrossModalTransformer(nn.Module):
    def __init__(self, num_layers: int = 4):
        super(CrossModalTransformer, self).__init__()
        self.num_layers = num_layers

    def forward(self, primary, secondary):
        for _ in range(self.num_layers):
            primary = CrossModalTransformerLayer(primary.shape, secondary.shape, 1024)(primary, secondary)
        return primary

