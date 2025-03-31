import math
import torch
import torch.nn as nn
from typing import List

class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = x.permute(1, 0, 2)
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
        assert x.shape == torch.Size([self.seq_len, self.feature_dim]), "Vector Shape is incorrect"
        return x + self.pe

class CrossModalTransformer(nn.Module):
    def __init__(self, primary_in_shape: List[int], secondary_in_shape: List[int]):
        super(CrossModalTransformer, self).__init__()
        self.primary_ln = nn.LayerNorm(primary_in_shape)
        self.secondary_ln = nn.LayerNorm(secondary_in_shape)

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        primary = self.primary_ln(primary)
        secondary = self.secondary_ln(secondary)
        return primary

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        return x


