import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    """3D Self-Attention module with multi-head attention"""
    def __init__(self, channels, heads):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)  # Normalize input features
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)  # Multi-head attention
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)  # Query/Key/Value projection
        self.out = nn.Conv3d(channels, channels, 1)  # Output projection

    def forward(self, x):
        B, C, D, H, W = x.shape  # Batch, Channels, Depth, Height, Width
        
        # Generate Q, K, V from input
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)  # Split into query, key, value
        
        # Reshape for attention: [B, C, D*H*W] -> [B, D*H*W, C]
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = v.view(B, C, -1).permute(0, 2, 1)

        # Compute attention
        attn, _ = self.mha(q, k, v)
        
        # Reshape back to 3D: [B, C, D, H, W]
        out = attn.permute(0, 2, 1).view(B, C, D, H, W)
        return x + self.out(out)  # Residual connection

class Block(nn.Module):
    """3D Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, stride, emb_dim=256):
        super().__init__()
        # Main 3D convolution path
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

        # Time embedding projection
        self.emb_layer = nn.Linear(emb_dim, out_channels)

        # Shortcut connection for residual
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1, stride)

    def forward(self, x, t):
        # Combine main path, shortcut, and time embedding
        return self.block(x) + self.shortcut(x) + self.emb_layer(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
class UNet3d(nn.Module):
    """3D U-Net architecture for video diffusion models"""
    def __init__(self, n_steps, emb_dim, channel_mult, attn_heads):
        super().__init__()
        self.n_steps = n_steps  # Total diffusion steps
        # Base channel configuration with multiplier
        _channels = (8, 16, 32, 64, 128)
        self.channels = [int(c*channel_mult) for c in _channels]
        self.attn_heads = attn_heads  # Number of attention heads

        # Input convolution (3 channels for RGB frames)
        self.input = nn.Conv3d(3, self.channels[0], 3, padding=1)

        # --- Encoder Path (Downsampling) ---
        # Downsample in spatial dimensions only (depth remains constant)
        self.down1 = Block(self.channels[0], self.channels[1], (1,2,2), emb_dim)  # Spatial downsampling
        self.down11 = Block(self.channels[1], self.channels[1], 1, emb_dim)        # Feature refinement
        
        self.down2 = Block(self.channels[1], self.channels[2], (1,2,2), emb_dim)  # Spatial downsampling
        self.down22 = Block(self.channels[2], self.channels[2], 1, emb_dim)        # Feature refinement
        
        self.down3 = Block(self.channels[2], self.channels[3], (1,2,2), emb_dim)  # Spatial downsampling
        self.down33 = Block(self.channels[3], self.channels[3], 1, emb_dim)        # Feature refinement
        
        self.down4 = Block(self.channels[3], self.channels[3], (1,2,2), emb_dim)  # Spatial downsampling
        self.down44 = Block(self.channels[3], self.channels[3], 1, emb_dim)        # Feature refinement

        # --- Bottleneck Path ---
        self.bottleneck1 = Block(self.channels[3], self.channels[4], 1, emb_dim)
        self.mid_attn1 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck2 = Block(self.channels[4], self.channels[4], 1, emb_dim)
        self.mid_attn2 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck3 = Block(self.channels[4], self.channels[3], emb_dim)
        self.mid_attn3 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()

        # --- Decoder Path (Upsampling) ---
        self.up1 = Block(self.channels[4], self.channels[2], 1, emb_dim)
        self.up11 = Block(self.channels[2], self.channels[2], 1, emb_dim)
        
        self.up2 = Block(self.channels[3], self.channels[1], 1, emb_dim)
        self.up22 = Block(self.channels[1], self.channels[1], 1, emb_dim)
        
        self.up3 = Block(self.channels[2], self.channels[0], 1, emb_dim)
        self.up33 = Block(self.channels[0], self.channels[0], 1, emb_dim)
        
        self.up4 = Block(self.channels[1], self.channels[0], 1, emb_dim)
        self.up44 = Block(self.channels[0], self.channels[0], 1, emb_dim)

        # Output convolution (removes temporal dimension)
        self.output = nn.Conv3d(self.channels[0], 3, 3, padding=(0, 1, 1))

        # Sinusoidal time embeddings
        self.embedding = self.sinusoidal_embeddings(n_steps, emb_dim).to(device)

    def forward(self, x, t):
        # Process time embedding
        t = self.embedding[t]
        
        # --- Encoder ---
        x1 = self.input(x)  # Initial convolution
        
        x2 = self.down1(x1, t)  # Downsample spatial dimensions
        x2 = self.down11(x2, t)  # Refine features
        
        x3 = self.down2(x2, t)  # Downsample spatial dimensions
        x3 = self.down22(x3, t)  # Refine features
        
        x4 = self.down3(x3, t)  # Downsample spatial dimensions
        x4 = self.down33(x4, t)  # Refine features
        
        x5 = self.down4(x4, t)  # Downsample spatial dimensions
        x5 = self.down44(x5, t)  # Refine features
        
        # --- Bottleneck ---
        x = self.bottleneck1(x5, t)
        x = self.mid_attn1(x)
        x = self.bottleneck2(x, t)
        x = self.mid_attn2(x)
        x = self.bottleneck3(x, t)
        x = self.mid_attn3(x)
        
        # --- Decoder with skip connections ---
        # Upsample to x4 resolution (trilinear for 3D)
        x = F.interpolate(x, size=(x4.shape[2:]), mode='trilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)  # Skip connection
        x = self.up1(x, t)
        x = self.up11(x, t)
        
        # Upsample to x3 resolution
        x = F.interpolate(x, size=(x3.shape[2:]), mode='trilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.up2(x, t)
        x = self.up22(x, t)
        
        # Upsample to x2 resolution
        x = F.interpolate(x, size=(x2.shape[2:]), mode='trilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.up3(x, t)
        x = self.up33(x, t)
        
        # Upsample to x1 resolution
        x = F.interpolate(x, size=(x1.shape[2:]), mode='trilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.up4(x, t)
        x = self.up44(x, t)
        
        # Final output convolution and remove temporal dimension
        x = self.output(x).squeeze(2)  # Output shape: [B, 3, H, W]
        return x
    
    def sinusoidal_embeddings(self, t, emb_dim=256):
        """Create sinusoidal position embeddings for timesteps"""
        denom = 10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim)
        positions = torch.arange(0, t).float().unsqueeze(1)
        embeddings = torch.zeros(t, emb_dim)
        embeddings[:, 0::2] = torch.sin(positions / denom)  # Even indices: sine
        embeddings[:, 1::2] = torch.cos(positions / denom)  # Odd indices: cosine
        return embeddings
