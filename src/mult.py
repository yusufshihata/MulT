import torch
import torch.nn as nn
import math

class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
             raise ValueError(f"Input tensor must have 3 dimensions (batch, seq_len, channels), got {x.dim()}")

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)

        return x

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

        self.q_proj = nn.Linear(dims, dims)
        self.kv_proj = nn.Linear(dims, dims * 2)

        self.is_first_layer = (primary_feature_dim != dims) or (secondary_feature_dim != dims)
        if self.is_first_layer:
            self.primary_proj_first = nn.Linear(primary_feature_dim, dims)
            self.secondary_proj_first = nn.Linear(secondary_feature_dim, dims * 2)


        self.attn_norm = nn.LayerNorm(dims)
        self.kv_norm = nn.LayerNorm(dims)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(self.head_dim)

        self.out_proj = nn.Linear(dims, dims)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims * 4, dims),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len_p, _ = primary.shape
        _, seq_len_s, _ = secondary.shape

        residual_p = primary

        if self.is_first_layer:
            primary_for_q = self.primary_proj_first(primary)
            residual_p = primary_for_q
            kv_input = self.secondary_proj_first(secondary)
        else:
            primary_for_q = primary
            kv_input = secondary

        q_proj_in = self.primary_proj(primary) if hasattr(self, 'primary_proj') else primary
        kv_proj_in = self.secondary_proj(secondary) if hasattr(self, 'secondary_proj') else secondary

        if hasattr(self, 'primary_proj_first'): # Check if we created specific projections
             q_in = self.primary_proj_first(primary)
             kv_source = self.secondary_proj_first(secondary)
        else:
             q_in = primary

        residual_p = primary
        q_potential = self.primary_proj(primary)
        if primary.shape[-1] == self.dims:
             residual_p = primary
        else:
             pass

        primary_norm = self.attn_norm(primary)
        secondary_norm = self.kv_norm(secondary)

        Q = self.q_proj(primary_norm)
        
        kv_proj_normalized = self.secondary_proj(secondary_norm)
        
        secondary_norm_original_dim = nn.LayerNorm(secondary_feature_dim).to(secondary.device)(secondary)
        kv_projected = self.secondary_proj(secondary_norm_original_dim)
        K, V = torch.split(kv_projected, self.dims, dim=-1)

        Q = Q.view(batch_size, seq_len_p, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_s, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_s, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-1, -2)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_p, self.dims)
        attn_output = self.out_proj(attn_output)

        primary_after_attn = primary + self.attn_dropout(attn_output)

        ffn_input = primary_after_attn
        ffn_norm_out = self.ffn_norm(ffn_input)
        ffn_out = self.ffn(ffn_norm_out)

        primary_final = ffn_input + self.ffn_dropout(ffn_out)

        return primary_final

class CrossModalTransformer(nn.Module):
    def __init__(self, primary_feature_dim: int, secondary_feature_dim: int, dims: int, num_heads: int, num_layers: int = 4, dropout: float = 0.1):
        super(CrossModalTransformer, self).__init__()
        self.num_layers = num_layers
        self.dims = dims

        self.primary_input_proj = nn.Linear(primary_feature_dim, dims)
        
        self.secondary_input_proj = nn.Linear(secondary_feature_dim, dims)

        self.initial_primary_norm = nn.LayerNorm(dims)
        self.initial_secondary_norm = nn.LayerNorm(dims)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(CrossModalTransformerLayer(
                dims=dims,
                num_heads=num_heads,
                primary_feature_dim=dims,
                secondary_feature_dim=dims,
                dropout=dropout
            ))

        self.final_norm = nn.LayerNorm(dims)


    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        primary_out = self.primary_input_proj(primary)
        secondary_proj = self.secondary_input_proj(secondary)


        for layer in self.layers:
            primary_out = layer(primary_out, secondary_proj)

        primary_out = self.final_norm(primary_out)

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

