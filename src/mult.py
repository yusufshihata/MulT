import torch
import torch.nn as nn
from typing import Optional

class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
    
    def forward(self, x):
        return self.conv1d(x)

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        return x

class CrossModalityAttention(nn.Module):
    def __init__(self):
        super(CrossModalityAttention, self).__init__()

    def forward(self, x):
        return x

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        return x

