import timm
import torch
import torch.nn as nn
from typing import Optional

class TinyViTMultiChannel:
    """
    Factory class to create TinyViT models modified for multi-channel input.
    """

    @staticmethod
    def create_model(
        num_channels: int = 15,
        model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k',
        pretrained: bool = True,
        num_classes: int = 1000
    ):
        """
        Creates a TinyViT model modified for custom number of input channels.

        Args:
            num_channels: Number of input channels (default: 15)
            model_name: TinyViT model variant from timm
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes

        Returns:
            Modified TinyViT model
        """
        # Load base TinyViT model
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        # Modify the patch embedding layer for multi-channel input
        TinyViTMultiChannel._modify_patch_embedding(model, num_channels)

        return model

    @staticmethod
    def _modify_patch_embedding(model, num_channels: int):
        """
        Modifies the patch embedding layer to accept multi-channel input.
        TinyViT uses conv1 and conv2 in patch_embed, not a single proj layer.
        """
        # Get the first convolution layer in patch embedding
        original_conv = model.patch_embed.conv1.conv  # ConvNorm has a conv attribute

        # Extract configuration
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        bias = original_conv.bias is not None

        # Create new first convolution layer
        new_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Initialize weights intelligently
        with torch.no_grad():
            if hasattr(original_conv, 'weight') and original_conv.weight is not None:
                # Average the original RGB weights across channels
                original_weight = original_conv.weight.clone()
                avg_weight = original_weight.mean(dim=1, keepdim=True)

                # Replicate averaged weights for all new channels
                new_weight = avg_weight.repeat(1, num_channels, 1, 1)
                new_conv.weight.copy_(new_weight)

                print(f"✅ Initialized patch embedding with averaged RGB weights")
            else:
                # Fallback to random initialization
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                print("⚠️ Using random initialization for patch embedding")

            # Copy bias if it exists
            if bias and hasattr(original_conv, 'bias') and original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # Replace the original convolution layer
        model.patch_embed.conv1.conv = new_conv

        print(f"✅ Modified TinyViT for {num_channels} input channels")
        return model

    @staticmethod
    def get_backbone_for_ssl(
        num_channels: int = 15,
        model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k',
        pretrained_path: Optional[str] = None,
        timm_pretrained: bool = True,
    ):
        """
        Creates a TinyViT backbone suitable for SSL training (without classifier head).

        Args:
            num_channels: Number of input channels
            model_name: TinyViT model variant

        Returns:
            TinyViT backbone model
        """
        # Create model without the final classification layer
        model = timm.create_model(
            model_name,
            pretrained=(timm_pretrained if pretrained_path is None else False),
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )

        # If a local pretrained checkpoint is provided, load it first
        if pretrained_path is not None:
            try:
                sd = torch.load(pretrained_path, map_location='cpu')
                # Some checkpoints save under 'state_dict' or have prefixes
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                model.load_state_dict(sd, strict=False)
                print(f"✅ Loaded local pretrained weights from {pretrained_path}")
            except Exception as e:
                print(f"⚠️ Failed to load local pretrained weights: {e}")

        # Modify patch embedding
        TinyViTMultiChannel._modify_patch_embedding(model, num_channels)

        return model


class TinyViTBackbone(nn.Module):
    """
    Wrapper for TinyViT backbone that ensures proper output dimensions for SSL.
    """

    def __init__(self, num_channels: int = 15, model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k', *, pretrained_path: Optional[str] = None, timm_pretrained: bool = True):
        super().__init__()

        # Create backbone
        self.backbone = TinyViTMultiChannel.get_backbone_for_ssl(
            num_channels=num_channels,
            model_name=model_name,
            pretrained_path=pretrained_path,
            timm_pretrained=timm_pretrained,
        )

        # Get the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, num_channels, 460, 460)
            dummy_output = self.backbone(dummy_input)

            # Handle different output shapes
            if len(dummy_output.shape) == 4:  # (B, C, H, W)
                self.feature_dim = dummy_output.shape[1]  # Use channel dimension
            elif len(dummy_output.shape) == 3:  # (B, N, D) - transformer output
                self.feature_dim = dummy_output.shape[-1]
            elif len(dummy_output.shape) == 2:  # (B, D) - already pooled
                self.feature_dim = dummy_output.shape[-1]
            else:
                raise ValueError(f"Unexpected backbone output shape: {dummy_output.shape}")

        print(f"✅ TinyViT backbone feature dimension: {self.feature_dim}")

    def forward(self, x):
        features = self.backbone(x)

        # Handle different output shapes and apply appropriate pooling
        if len(features.shape) == 4:  # (B, C, H, W)
            features = features.mean(dim=[-2, -1])  # Global average pooling
        elif len(features.shape) == 3:  # (B, N, D) - transformer tokens
            features = features.mean(dim=1)  # Average over sequence dimension
        # If already (B, D), no pooling needed

        return features


def create_ssl_model_components(
    num_channels: int = 15,
    model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k',
    projection_dim: int = 128,
    hidden_dim: Optional[int] = None,
    *,
    pretrained_path: Optional[str] = None,
    timm_pretrained: bool = True,
):
    """
    Creates backbone and projection head for SSL training.

    Args:
        num_channels: Number of input channels
        model_name: TinyViT variant to use
        projection_dim: Output dimension of projection head
        hidden_dim: Hidden dimension of projection head (defaults to feature_dim)

    Returns:
        tuple: (backbone, projection_head)
    """
    backbone = TinyViTBackbone(num_channels, model_name, pretrained_path=pretrained_path, timm_pretrained=timm_pretrained)

    if hidden_dim is None:
        hidden_dim = backbone.feature_dim

    from lightly.models.modules import SimCLRProjectionHead
    projection_head = SimCLRProjectionHead(
        input_dim=backbone.feature_dim,
        hidden_dim=hidden_dim,
        output_dim=projection_dim
    )

    return backbone, projection_head
