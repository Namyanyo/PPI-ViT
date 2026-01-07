"""
CNN and Vision Transformer Models for PPI Binary Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PPIClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        CNN-based classifier for PPI prediction

        Args:
            num_classes: Number of output classes (default: 2 for binary)
            dropout_rate: Dropout rate for regularization
        """
        super(PPIClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            out: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Convert image into patches and embed them

        Args:
            image_size: Size of input image
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, image_size, image_size)
        Returns:
            patches: (batch_size, n_patches, embed_dim)
        """
        x = self.projection(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        """
        Multi-head self-attention mechanism

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches, embed_dim)
        Returns:
            out: (batch_size, n_patches, embed_dim)
        """
        batch_size, n_patches, embed_dim = x.shape
        qkv = self.qkv(x) 
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                       three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        """
        Transformer encoder block

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches, embed_dim)
        Returns:
            out: (batch_size, n_patches, embed_dim)
        """
        x = x + self.attn(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


class PPIViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        """
        Vision Transformer for PPI classification

        Args:
            image_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
        """
        super(PPIViT, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embedding.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

  
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, image_size, image_size)
        Returns:
            out: (batch_size, num_classes)
        """
        batch_size = x.shape[0]

 
        x = self.patch_embedding(x) 

        # Prepend class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)  
   
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

     
        for block in self.transformer_blocks:
            x = block(x)


        x = self.norm(x)
        cls_token_final = x[:, 0] 
        out = self.head(cls_token_final) 

        return out


class PPIViTSmall(nn.Module):
    def __init__(self, image_size=224, num_classes=2, dropout=0.1):
        """
        Smaller Vision Transformer for PPI classification (faster training)

        Args:
            image_size: Input image size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(PPIViTSmall, self).__init__()

       
        self.vit = PPIViT(
            image_size=image_size,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            dropout=dropout
        )

    def forward(self, x):
        return self.vit(x)


class PPIViTBase(nn.Module):
    def __init__(self, image_size=224, num_classes=2, dropout=0.1):
        """
        Base Vision Transformer for PPI classification (ViT-B/16 config)

        Args:
            image_size: Input image size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(PPIViTBase, self).__init__()

       
        self.vit = PPIViT(
            image_size=image_size,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=dropout
        )

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
   main()

