import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()        
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = v.view(B, C, -1).permute(0, 2, 1)

        attn, _ = self.mha(q, k, v)
        out = attn.permute(0, 2, 1).view(B, C, H, W)
        return x + self.out(out)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.emb_layer = nn.Linear(emb_dim, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x, t):
        return self.block(x) + self.shortcut(x) + self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)
    
class UNet(nn.Module):
    def __init__(self, n_steps, emb_dim=256, channel_mult=1, attn_heads=0, in_channels=9):
        super().__init__()
        self.n_steps = n_steps
        _channels = (8, 16, 32, 64, 128)
        self.channels = [int(c*channel_mult) for c in _channels]
        self.attn_heads = attn_heads

        self.input = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)

        self.down1 = Block(self.channels[0], self.channels[1], 2, emb_dim)
        # self.down_attn1 = SelfAttention(self.channels[1], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.down2 = Block(self.channels[1], self.channels[2], 2, emb_dim)
        # self.down_attn2 = SelfAttention(self.channels[2], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.down3 = Block(self.channels[2], self.channels[3], 2, emb_dim)
        # self.down_attn3 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.down4 = Block(self.channels[3], self.channels[3], 2, emb_dim)
        self.down_attn4 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()

        self.bottleneck1 = Block(self.channels[3], self.channels[4], 1, emb_dim)
        self.mid_attn1 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck2 = Block(self.channels[4], self.channels[4], 1, emb_dim)
        self.mid_attn2 = SelfAttention(self.channels[4], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.bottleneck3 = Block(self.channels[4], self.channels[3], 1, emb_dim)
        self.mid_attn3 = SelfAttention(self.channels[3], self.attn_heads) if attn_heads > 0 else nn.Identity()

        self.up1 = Block(self.channels[4], self.channels[2], 1, emb_dim)
        self.up_attn1 = SelfAttention(self.channels[2], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.up2 = Block(self.channels[3], self.channels[1], 1, emb_dim)
        # self.up_attn2 = SelfAttention(self.channels[1], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.up3 = Block(self.channels[2], self.channels[0], 1, emb_dim)
        # self.up_attn3 = SelfAttention(self.channels[0], self.attn_heads) if attn_heads > 0 else nn.Identity()
        self.up4 = Block(self.channels[1], self.channels[0], 1, emb_dim)
        # self.up_attn4 = SelfAttention(self.channels[0], self.attn_heads) if attn_heads > 0 else nn.Identity()

        self.output = nn.Conv2d(self.channels[0], 3, 3, padding=1)

        self.embedding = self.sinusoidal_embeddings(n_steps, emb_dim).to(device)

    def forward(self, x, t):
        t = self.embedding[t]
        x1 = self.input(x) 

        x2 = self.down1(x1, t)
        # x2 = self.down_attn1(x2)
        
        x3 = self.down2(x2, t)
        # x3 = self.down_attn2(x3)
        
        x4 = self.down3(x3, t)
        # x4 = self.down_attn3(x4)

        x5 = self.down4(x4, t)
        x5 = self.down_attn4(x5)
        
        x = self.bottleneck1(x5, t)
        x = self.mid_attn1(x)
        x = self.bottleneck2(x, t)
        x = self.mid_attn2(x)
        x = self.bottleneck3(x, t)
        x = self.mid_attn3(x)

        x = F.interpolate(x, size=(x4.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x, t)
        x = self.up_attn1(x)

        x = F.interpolate(x, size=(x3.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x, t)
        # x = self.up_attn2(x)

        x = F.interpolate(x, size=(x2.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x, t)
        # x = self.up_attn3(x)

        x = F.interpolate(x, size=(x1.shape[2:]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x, t)
        # x = self.up_attn4(x)

        x = self.output(x)

        return x
    
    def sinusoidal_embeddings(self, t, emb_dim=256):
        denom = 10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim)
        positions = torch.arange(0, t).float().unsqueeze(1)
        embeddings = torch.zeros(t, emb_dim)
        embeddings[:, 0::2] = torch.sin(positions / denom)
        embeddings[:, 1::2] = torch.cos(positions / denom)
        return embeddings