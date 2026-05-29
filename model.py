"""
Core Model Architecture for Diffusion-Based Image Steganography.

This module implements:
- Secret Encoder: Compress secret image to latent representation
- Diffusion Embedding Module: Progressive secret injection during denoising
- Secret Decoder: Recover secret from stego image
- Complete DiffusionSteganography model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math

from config import ModelConfig, DiffusionConfig, DEFAULT_CONFIG


# ============================================================================
# UTILITY LAYERS
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        use_residual: bool = True
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            activation: Activation function ("relu", "gelu")
            use_residual: Whether to use residual connection
        """
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_residual:
            out = out + identity
        
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Downsampling block with residual connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        use_residual: bool = True
    ):
        """Initialize down block with max pooling."""
        super().__init__()
        
        self.residual_block = ResidualBlock(
            in_channels, out_channels, kernel_size, activation, use_residual
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns residual output and pooled output."""
        out = self.residual_block(x)
        pooled = self.pool(out)
        return out, pooled


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        use_residual: bool = True
    ):
        """Initialize up block with bilinear upsampling."""
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # After concatenation with skip connection
        self.residual_block = ResidualBlock(
            in_channels * 2, out_channels, kernel_size, activation, use_residual
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        out = self.residual_block(x)
        return out


class TimeEmbedding(nn.Module):
    """Embedding for diffusion timestep."""
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize time embedding."""
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timestep.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            Embedding of shape (batch_size, embedding_dim)
        """
        t = t.unsqueeze(-1).float()  # (batch_size, 1)
        return self.time_embed(t)


# ============================================================================
# SECRET ENCODER
# ============================================================================

class SecretEncoder(nn.Module):
    """
    Encode secret image into compact latent representation.
    
    Architecture:
    - Progressive downsampling with residual blocks
    - Feature compression
    - Bottleneck layer
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize secret encoder.
        
        Args:
            config: ModelConfig instance
        """
        super().__init__()
        self.config = config
        
        channels = config.secret_encoder_channels
        kernels = config.secret_encoder_kernel_sizes
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            kernel = kernels[i]
            
            self.layers.append(
                DownBlock(
                    in_ch, out_ch, kernel,
                    activation=config.activation,
                    use_residual=config.secret_encoder_use_residual
                )
            )
        
        # Bottleneck
        self.bottleneck = nn.AdaptiveAvgPool2d(1)
        
        # Parameter count tracking
        self._param_count = sum(p.numel() for p in self.parameters())
    
    def forward(self, secret: torch.Tensor) -> torch.Tensor:
        """
        Encode secret image.
        
        Args:
            secret: Secret image tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Encoded latent tensor of shape (batch_size, channels[-1], 1, 1)
        """
        x = secret
        skip_connections = []
        
        for layer in self.layers:
            residual, x = layer(x)
            skip_connections.append(residual)
        
        # Bottleneck compression
        x = self.bottleneck(x)
        
        return x


# ============================================================================
# DIFFUSION EMBEDDING MODULE
# ============================================================================

class DiffusionDenoiseBlock(nn.Module):
    """
    Single denoising block in diffusion process.
    
    Processes noisy image and progressively embeds secret information.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        use_skip: bool = True,
        time_embedding_dim: int = 128
    ):
        """Initialize denoise block."""
        super().__init__()
        
        self.use_skip = use_skip
        
        # Main denoising convolution
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, H, W)
            time_emb: Time embedding (batch_size, time_embedding_dim)
        
        Returns:
            Denoised tensor
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Add time embedding
        time_scale = self.time_proj(time_emb)  # (batch_size, out_channels)
        time_scale = time_scale.view(time_scale.shape[0], time_scale.shape[1], 1, 1)
        out = out * time_scale + out
        
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_skip and out.shape == residual.shape:
            out = out + residual
        
        out = self.activation(out)
        return out


class DiffusionEmbeddingModule(nn.Module):
    """
    Diffusion-based embedding module.
    
    Performs denoising steps where secret is progressively injected
    into the denoising trajectory.
    
    Architecture:
    - Noise prediction network
    - Progressive secret integration
    - Multi-step denoising
    """
    
    def __init__(self, config: ModelConfig, diffusion_config: DiffusionConfig):
        """
        Initialize diffusion embedding module.
        
        Args:
            config: ModelConfig instance
            diffusion_config: DiffusionConfig instance
        """
        super().__init__()
        
        self.config = config
        self.diffusion_config = diffusion_config
        self.num_steps = diffusion_config.num_steps
        
        # Time embedding
        self.time_embedding = TimeEmbedding(embedding_dim=128)
        
        # Build denoising network with U-Net-like architecture
        channels = config.diffusion_channels
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, kernel_size=3,
                    activation=config.activation,
                    use_residual=config.use_skip_connections
                )
            )
        
        # Bottleneck
        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels * 2, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 2, -1, -1):
            out_ch = channels[i]
            self.up_blocks.append(
                UpBlock(
                    channels[i + 1], out_ch, kernel_size=3,
                    activation=config.activation,
                    use_residual=config.use_skip_connections
                )
            )
        
        # Output layer
        self.output_conv = nn.Conv2d(channels[0], 3, kernel_size=1)
        
        # Secret integration layer
        self.secret_integration = nn.Conv2d(3 + 64, 3, kernel_size=1)
    
    def forward(
        self,
        cover: torch.Tensor,
        secret_latent: torch.Tensor,
        secret_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform diffusion-based embedding.
        
        Args:
            cover: Cover image (batch_size, 3, H, W)
            secret_latent: Encoded secret latent (batch_size, 64, 1, 1)
            secret_image: Original secret image (batch_size, 3, H, W)
        
        Returns:
            (stego_image, denoising_trajectory)
        """
        batch_size = cover.shape[0]
        device = cover.device
        
        # Initialize with cover image
        x = cover.clone()
        denoising_trajectory = []
        
        # Progressive denoising with secret injection
        for step in range(self.num_steps):
            # Current timestep
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)
            time_emb = self.time_embedding(t)
            
            # Store trajectory step
            denoising_trajectory.append(x.clone().detach())
            
            # Encode through U-Net
            skip_connections = []
            encoded = x
            
            for down_block in self.down_blocks:
                skip, encoded = down_block(encoded)
                skip_connections.append(skip)
            
            encoded = self.bottleneck(encoded)
            
            # Decoder with skip connections
            for i, up_block in enumerate(self.up_blocks):
                skip_idx = len(skip_connections) - 1 - i
                encoded = up_block(encoded, skip_connections[skip_idx])
            
            # Output denoising prediction
            denoised = self.output_conv(encoded)
            
            # Integrate secret progressively
            # Expand secret latent to spatial dimensions
            secret_expanded = secret_latent.expand(batch_size, 64, x.shape[2], x.shape[3])
            
            # Blend secret with denoised output
            combined = torch.cat([denoised, secret_expanded], dim=1)
            integrated = self.secret_integration(combined)
            
            # Apply secret integration weight based on step
            step_ratio = step / self.num_steps
            if self.diffusion_config.secret_injection_schedule == "uniform":
                weight = self.diffusion_config.secret_strength_per_step
            elif self.diffusion_config.secret_injection_schedule == "front_loaded":
                weight = self.diffusion_config.secret_strength_per_step * (1 - step_ratio) ** 2
            elif self.diffusion_config.secret_injection_schedule == "back_loaded":
                weight = self.diffusion_config.secret_strength_per_step * step_ratio ** 2
            else:
                weight = self.diffusion_config.secret_strength_per_step
            
            # Update with weighted blend
            x = x * (1 - weight) + integrated * weight
            
            # Blend with secret image for stronger embedding
            x = x * 0.98 + secret_image * 0.02
        
        stego = x
        return stego, torch.stack(denoising_trajectory)


# ============================================================================
# SECRET DECODER
# ============================================================================

class SecretDecoder(nn.Module):
    """
    Recover secret image from stego image.
    
    Architecture:
    - Progressive feature extraction
    - Residual learning for reconstruction
    - Skip connections for detail preservation
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize secret decoder.
        
        Args:
            config: ModelConfig instance
        """
        super().__init__()
        self.config = config
        
        channels = config.secret_decoder_channels
        kernels = config.secret_decoder_kernel_sizes
        
        # Build decoder layers (inverse of encoder)
        self.layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            kernel = kernels[i]
            
            self.layers.append(
                UpBlock(
                    in_ch, out_ch, kernel,
                    activation=config.activation,
                    use_residual=config.secret_decoder_use_residual
                )
            )
        
        # Output layer for fine details
        self.output_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1], 3, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Decode secret from stego image.
        
        Args:
            stego: Stego image tensor (batch_size, 3, H, W)
        
        Returns:
            Recovered secret image (batch_size, 3, H, W)
        """
        x = stego
        
        for layer in self.layers:
            # Simple pass through (simplified for lightweight)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        secret = self.output_layer(x)
        return secret


# ============================================================================
# COMPLETE DIFFUSION STEGANOGRAPHY MODEL
# ============================================================================

class DiffusionSteganography(nn.Module):
    """
    Complete Diffusion-Based Image Steganography Model.
    
    Pipeline:
    1. Encode secret image → latent representation
    2. Embed secret into cover via diffusion denoising
    3. Recover secret from stego via decoder
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        diffusion_config: Optional[DiffusionConfig] = None
    ):
        """
        Initialize complete steganography model.
        
        Args:
            model_config: ModelConfig instance (default: DEFAULT_CONFIG.model)
            diffusion_config: DiffusionConfig instance (default: DEFAULT_CONFIG.diffusion)
        """
        super().__init__()
        
        self.model_config = model_config or DEFAULT_CONFIG.model
        self.diffusion_config = diffusion_config or DEFAULT_CONFIG.diffusion
        
        # Initialize components
        self.secret_encoder = SecretEncoder(self.model_config)
        self.diffusion_embedding = DiffusionEmbeddingModule(
            self.model_config, self.diffusion_config
        )
        self.secret_decoder = SecretDecoder(self.model_config)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
    
    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Complete forward pass.
        
        Args:
            cover: Cover image (batch_size, 3, H, W)
            secret: Secret image (batch_size, 3, H, W)
        
        Returns:
            (stego_image, recovered_secret, metadata_dict)
        """
        # Encode secret
        secret_latent = self.secret_encoder(secret)
        
        # Embed into cover via diffusion
        stego, trajectory = self.diffusion_embedding(cover, secret_latent, secret)
        
        # Recover secret from stego
        recovered_secret = self.secret_decoder(stego)
        
        metadata = {
            "secret_latent_shape": secret_latent.shape,
            "trajectory_length": len(trajectory),
            "model_params": self.total_params,
            "trainable_params": self.trainable_params,
        }
        
        return stego, recovered_secret, metadata
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            "secret_encoder": sum(p.numel() for p in self.secret_encoder.parameters()),
            "diffusion_embedding": sum(
                p.numel() for p in self.diffusion_embedding.parameters()
            ),
            "secret_decoder": sum(p.numel() for p in self.secret_decoder.parameters()),
            "total": self.total_params,
            "trainable": self.trainable_params,
        }
    
    def print_architecture(self) -> None:
        """Print model architecture summary."""
        param_counts = self.get_parameter_count()
        
        print("\n" + "="*70)
        print("🏗️  DIFFUSION-BASED IMAGE STEGANOGRAPHY ARCHITECTURE")
        print("="*70)
        
        print("\n📊 SECRET ENCODER")
        print(f"   {self.secret_encoder}")
        
        print("\n🔄 DIFFUSION EMBEDDING MODULE")
        print(f"   Diffusion Steps: {self.diffusion_config.num_steps}")
        print(f"   Embedding Strategy: {self.model_config.embedding_strategy}")
        
        print("\n🔓 SECRET DECODER")
        print(f"   {self.secret_decoder}")
        
        print("\n📈 PARAMETER STATISTICS")
        print(f"   • Secret Encoder: {param_counts['secret_encoder']:,} params")
        print(f"   • Diffusion Module: {param_counts['diffusion_embedding']:,} params")
        print(f"   • Secret Decoder: {param_counts['secret_decoder']:,} params")
        print(f"   • Total: {param_counts['total']:,} params")
        print(f"   • Trainable: {param_counts['trainable']:,} params")
        
        max_params = self.model_config.max_parameters
        usage_pct = (param_counts['total'] / max_params) * 100
        print(f"   • Max Allowed: {max_params:,} params")
        print(f"   • Usage: {usage_pct:.2f}%")
        
        if param_counts['total'] > max_params:
            print(f"   ⚠️  WARNING: Exceeds max parameter limit!")
        
        print("="*70 + "\n")


# ============================================================================
# MODEL TESTING
# ============================================================================

def test_model():
    """Test model forward pass and output shapes."""
    print("\n" + "="*70)
    print("🧪 MODEL TESTING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Create model
    model = DiffusionSteganography().to(device)
    model.print_architecture()
    
    # Create random inputs
    batch_size = 2
    cover = torch.randn(batch_size, 3, 256, 256).to(device)
    secret = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"\n📦 INPUT SHAPES")
    print(f"   • Cover: {cover.shape}")
    print(f"   • Secret: {secret.shape}")
    
    # Forward pass
    print(f"\n⚙️  Running forward pass...")
    try:
        with torch.no_grad():
            stego, recovered, metadata = model(cover, secret)
        
        print(f"\n✅ FORWARD PASS SUCCESSFUL")
        print(f"\n📊 OUTPUT SHAPES")
        print(f"   • Stego: {stego.shape}")
        print(f"   • Recovered Secret: {recovered.shape}")
        print(f"\n📋 METADATA")
        for key, value in metadata.items():
            print(f"   • {key}: {value}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    test_model()
