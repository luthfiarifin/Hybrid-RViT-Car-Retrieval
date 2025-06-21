import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class HybridRestnetVit(nn.Module):
    """
    HybridResnetViT: A hybrid architecture combining ResNet-50 and Vision Transformer (ViT).

    - Uses a pretrained ResNet-50 as a feature extractor (removes avgpool and fc layers).
    - Converts the ResNet feature map into patches using a Conv2d layer (patch embedding).
    - Adds a learnable [CLS] token and positional embeddings to the patch sequence.
    - Processes the sequence with a Transformer encoder.
    - Uses the [CLS] token output for final classification.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=7,
        num_classes=8,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
    ):
        """
        Args:
            image_size (int): Input image size (e.g., 224 for 224x224 images).
            patch_size (int): Patch size for patch embedding (applied to feature map).
            num_classes (int): Number of output classes.
            embed_dim (int): Embedding dimension for Transformer.
            num_heads (int): Number of attention heads in Transformer.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # 1. CNN Backbone: Pretrained ResNet-50 (remove avgpool and fc)
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # For 224x224 input, output feature map is (batch, 2048, 7, 7)

        # 2. Patch Embedding: Convert feature map to patch sequence
        # The number of patches is (feature_map_size // patch_size) ** 2
        feature_map_size = image_size // 32  # 224/32=7
        self.num_patches = (feature_map_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=2048,  # ResNet-50 output channels
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 3. [CLS] Token: Learnable token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 4. Positional Embedding: Learnable position encoding for [CLS] + patches
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)

        # 5. Transformer Encoder: Sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 6. Classification Head: Predict class from [CLS] token output
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Step 1: Extract feature map from ResNet-50 backbone
        # Input: (batch, 3, 224, 224) -> Output: (batch, 2048, 7, 7)
        feature_map = self.backbone(x)

        # Step 2: Patch embedding (Conv2d)
        # Input: (batch, 2048, 7, 7) -> Output: (batch, embed_dim, 1, 1)
        patches = self.patch_embed(feature_map)

        # Step 3: Flatten patches to sequence (batch, num_patches, embed_dim)
        patches = patches.flatten(2).transpose(1, 2)

        # Step 4: Prepend [CLS] token to patch sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)

        # Step 5: Add positional embeddings and apply dropout
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        # Step 6: Transformer encoder
        transformer_output = self.transformer_encoder(x)

        # Step 7: Use [CLS] token output for classification
        cls_output = transformer_output[:, 0]
        logits = self.classifier(cls_output)
        return logits
