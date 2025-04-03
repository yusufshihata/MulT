import math
import torch
import torch.nn as nn

class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 2)
        x = self.conv1d(x)
        return x.permute(1, 0, 2).squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, seq_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, feature_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim must be even, got {feature_dim}")
            
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
             raise ValueError(f"Input tensor must have 3 dimensions (batch, seq, feature), got {x.dim()}")
             
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CrossModalTransformerLayer(nn.Module):
    def __init__(self, dims: int, num_heads: int, primary_feature_dim: int, secondary_feature_dim: int, dropout: float = 0.1):
        super(CrossModalTransformerLayer, self).__init__()
        
        if dims % num_heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by num_heads ({num_heads})")

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads

        self.primary_ln = nn.LayerNorm(dims)
        self.secondary_ln = nn.LayerNorm(dims)

        self.primary_proj = nn.Linear(primary_feature_dim, dims)
        self.secondary_proj = nn.Linear(secondary_feature_dim, dims * 2)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(self.head_dim)

        self.out_proj = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims * 4, dims),
            nn.Dropout(dropout)
        )


    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len_p, _ = primary.shape
        _, seq_len_s, _ = secondary.shape

        q_proj = self.primary_proj(primary)
        
        kv_proj = self.secondary_proj(secondary)
        k_proj, v_proj = torch.split(kv_proj, self.dims, dim=-1)

        q_norm = self.primary_ln(q_proj)
        k_norm = self.secondary_ln(k_proj)
        v_norm = self.secondary_ln(v_proj)

        residual_q = q_proj

        Q = q_norm.view(batch_size, seq_len_p, self.num_heads, self.head_dim).transpose(1, 2)

        K = k_norm.view(batch_size, seq_len_s, self.num_heads, self.head_dim).transpose(1, 2)

        V = v_norm.view(batch_size, seq_len_s, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-1, -2)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ V

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_p, self.dims)

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        primary_after_attn = residual_q + attn_output

        ffn_output = self.ffn(primary_after_attn)
        
        primary_final = primary_after_attn + ffn_output

        return primary_final


class CrossModalTransformer(nn.Module):
    def __init__(self, primary_feature_dim: int, secondary_feature_dim: int, dims: int, num_heads: int, num_layers: int = 4, dropout: float = 0.1):
        super(CrossModalTransformer, self).__init__()
        self.num_layers = num_layers
        self.dims = dims

        self.layers = nn.ModuleList([])
        self.layers.append(CrossModalTransformerLayer(dims, num_heads, primary_feature_dim, secondary_feature_dim, dropout=dropout))
        for _ in range(num_layers - 1):
            self.layers.append(CrossModalTransformerLayer(dims, num_heads, dims, secondary_feature_dim, dropout=dropout))

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        primary_out = primary
        for layer in self.layers:
            primary_out = layer(primary_out, secondary)

        return primary_out

# --- Example Usage ---
if __name__ == '__main__':
    # Parameters
    batch_size = 4
    seq_len_p = 20 # Sequence length for primary modality
    seq_len_s = 30 # Sequence length for secondary modality
    primary_feat_dim = 512 # Input feature dimension for primary
    secondary_feat_dim = 768 # Input feature dimension for secondary
    model_dims = 1024 # Internal dimension of the transformer model
    num_heads = 8    # Number of attention heads
    num_layers = 6   # Number of transformer layers

    # Create dummy input tensors
    primary_input = torch.rand(batch_size, seq_len_p, primary_feat_dim)
    secondary_input = torch.rand(batch_size, seq_len_s, secondary_feat_dim)
    transformer = CrossModalTransformer(
        primary_feature_dim=primary_feat_dim,
        secondary_feature_dim=secondary_feat_dim,
        dims=model_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )

    output = transformer(primary_input, secondary_input)

    print(f"Input primary shape: {primary_input.shape}")
    print(f"Input secondary shape: {secondary_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == torch.Size([batch_size, seq_len_p, model_dims])
    print("Multi-head cross-modal transformer ran successfully!")

    pe_seq_len = 50
    pe_feat_dim = 128
    pos_encoder = PositionalEncoding(feature_dim=pe_feat_dim, seq_len=pe_seq_len)
    test_input = torch.zeros(batch_size, pe_seq_len, pe_feat_dim)
    pe_output = pos_encoder(test_input)
    print(f"\nPositional Encoding Input Shape: {test_input.shape}")
    print(f"Positional Encoding Output Shape: {pe_output.shape}")
    assert pe_output.shape == test_input.shape
    print("Positional Encoding works with batch dimension.")

