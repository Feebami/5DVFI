import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    """Self-Attention module with multi-head attention"""
    def __init__(self, channels, heads):
        super().__init__()        
        self.norm = nn.GroupNorm(8, channels)  # Normalize input features
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)  # Multi-head attention
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)  # Query/Key/Value projection
        self.out = nn.Conv2d(channels, channels, 1)  # Output projection

    def forward(self, x):
        B, C, H, W = x.shape  # Batch, Channels, Height, Width
        
        # Generate Q, K, V from input
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)  # Split into query, key, value
        
        # Reshape for attention: [B, C, H*W] -> [B, H*W, C]
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = v.view(B, C, -1).permute(0, 2, 1)

        # Compute attention and combine with residual connection
        attn, _ = self.mha(q, k, v)
        out = attn.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to [B, C, H, W]
        return x + self.out(out)  # Residual connection

class Block(nn.Module):
    """Basic residual block with time embedding"""
    def __init__(self, in_channels, out_channels, stride, emb_dim=256):
        super().__init__()
        # Main convolution path
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # Time embedding projection
        self.emb_layer = nn.Linear(emb_dim, out_channels)

        # Shortcut connection for residual
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x, t):
        # Combine main path, shortcut, and time embedding
        return self.block(x) + self.shortcut(x) + self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)
    
class UNet(nn.Module):
    """U-Net architecture for diffusion models with optional attention"""
    def __init__(self, n_steps, emb_dim=256, channel_mult=1, attn_heads=0, in_channels=9):
        super().__init__()
        self.n_steps = n_steps  # Total diffusion steps
        # Base channel configuration with multiplier
        _channels = (8, 16, 32, 64, 128)
        self.channels = [int(c*channel_mult) for c in _channels]
        self.attn_heads = attn_heads  # Number of attention heads

        # Input convolution
        self.input = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)

        # --- Encoder Path ---
        self.down1 = Block(self.channels[0], self.channels[1], 2, emb_dim)
        self.down2 = Block(self.channels[1], self.channels[2], 2, emb_dim)
        self.down3 = Block(self.channels[2], self.channels[3], 2, emb_dim)
        self.down4 = Block(self.channels[3], self.channels[3], 2, emb_dim)
        self.down_attn4 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()

        # --- Bottleneck Path ---
        self.bottleneck1 = Block(self.channels[3], self.channels[4], 1, emb_dim)
        self.mid_attn1 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck2 = Block(self.channels[4], self.channels[4], 1, emb_dim)
        self.mid_attn2 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck3 = Block(self.channels[4], self.channels[3], 1, emb_dim)
        self.mid_attn3 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()

        # --- Decoder Path ---
        self.up1 = Block(self.channels[4], self.channels[2], 1, emb_dim)  # Channels: 64+64=128 -> 32
        self.up_attn1 = SelfAttention(self.channels[2], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.up2 = Block(self.channels[3], self.channels[1], 1, emb_dim)  # Channels: 32+32=64 -> 16
        self.up3 = Block(self.channels[2], self.channels[0], 1, emb_dim)  # Channels: 16+16=32 -> 8
        self.up4 = Block(self.channels[1], self.channels[0], 1, emb_dim)   # Channels: 8+8=16 -> 8

        # Output convolution
        self.output = nn.Conv2d(self.channels[0], 3, 3, padding=1)

        # Sinusoidal time embeddings
        self.embedding = self.sinusoidal_embeddings(n_steps, emb_dim).to(device)

    def forward(self, x, t):
        # Process time embedding
        t = self.embedding[t]
        
        # --- Encoder ---
        x1 = self.input(x)  # Initial convolution
        
        x2 = self.down1(x1, t)  # Downsample 2x
        x3 = self.down2(x2, t)  # Downsample 2x
        x4 = self.down3(x3, t)  # Downsample 2x
        x5 = self.down4(x4, t)  # Downsample 2x
        x5 = self.down_attn4(x5)  # Optional attention
        
        # --- Bottleneck ---
        x = self.bottleneck1(x5, t)
        x = self.mid_attn1(x)
        x = self.bottleneck2(x, t)
        x = self.mid_attn2(x)
        x = self.bottleneck3(x, t)
        x = self.mid_attn3(x)
        
        # --- Decoder with skip connections ---
        # Upsample to x4 resolution
        x = F.interpolate(x, size=(x4.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)  # Skip connection
        x = self.up1(x, t)
        x = self.up_attn1(x)
        
        # Upsample to x3 resolution
        x = F.interpolate(x, size=(x3.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.up2(x, t)
        
        # Upsample to x2 resolution
        x = F.interpolate(x, size=(x2.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.up3(x, t)
        
        # Upsample to x1 resolution
        x = F.interpolate(x, size=(x1.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.up4(x, t)
        
        # Final output convolution
        x = self.output(x)
        return x
    
    def sinusoidal_embeddings(self, t, emb_dim=256):
        """Create sinusoidal position embeddings for timesteps"""
        denom = 10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim)
        positions = torch.arange(0, t).float().unsqueeze(1)
        embeddings = torch.zeros(t, emb_dim)
        embeddings[:, 0::2] = torch.sin(positions / denom)  # Even indices: sine
        embeddings[:, 1::2] = torch.cos(positions / denom)  # Odd indices: cosine
        return embeddings
