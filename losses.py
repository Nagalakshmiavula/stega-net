"""
Loss Functions for Diffusion-Based Image Steganography.

This module implements:
- Cover reconstruction loss (L2)
- Secret reconstruction loss (L2)
- SSIM loss (structural similarity)
- Perceptual loss (VGG-based)
- Diffusion consistency loss
- Edge preservation loss
- Hybrid weighted loss combiner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional
import numpy as np

from config import TrainingConfig, DEFAULT_CONFIG


# ============================================================================
# BASIC LOSS FUNCTIONS
# ============================================================================

class L2Loss(nn.Module):
    """L2 (MSE) loss."""
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize L2 loss.
        
        Args:
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
        
        Returns:
            L2 loss value
        """
        diff = pred - target
        loss = torch.mean(diff ** 2) if self.reduction == "mean" else torch.sum(diff ** 2)
        return loss


class L1Loss(nn.Module):
    """L1 (MAE) loss."""
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize L1 loss.
        
        Args:
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss."""
        diff = torch.abs(pred - target)
        loss = torch.mean(diff) if self.reduction == "mean" else torch.sum(diff)
        return loss


# ============================================================================
# STRUCTURAL SIMILARITY (SSIM) LOSS
# ============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) loss.
    
    SSIM measures perceived quality better than MSE.
    Loss = 1 - SSIM (to minimize)
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        """
        Initialize SSIM loss.
        
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian kernel
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        
        # Create Gaussian kernel
        kernel_range = torch.arange(window_size).float() - (window_size - 1) / 2.0
        gaussian = torch.exp(-kernel_range.pow(2.0) / (2 * sigma ** 2))
        kernel_1d = gaussian / gaussian.sum()
        kernel_2d = kernel_1d.unsqueeze(-1) * kernel_1d.unsqueeze(0)
        
        self.register_buffer("kernel_2d", kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def _gaussian_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian filter."""
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        
        kernel = self.kernel_2d.expand(b * c, 1, -1, -1).to(x.device)
        x = F.conv2d(x, kernel, padding=self.window_size // 2, groups=b * c)
        
        return x.view(b, c, h, w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            SSIM loss (1 - SSIM)
        """
        # Constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Means
        mu_pred = self._gaussian_filter(pred)
        mu_target = self._gaussian_filter(target)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Variances and covariance
        sigma_pred_sq = self._gaussian_filter(pred ** 2) - mu_pred_sq
        sigma_target_sq = self._gaussian_filter(target ** 2) - mu_target_sq
        sigma_pred_target = self._gaussian_filter(pred * target) - mu_pred_target
        
        # SSIM formula
        numerator1 = 2 * mu_pred_target + c1
        numerator2 = 2 * sigma_pred_target + c2
        denominator1 = mu_pred_sq + mu_target_sq + c1
        denominator2 = sigma_pred_sq + sigma_target_sq + c2
        
        ssim = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        # Loss is 1 - SSIM
        loss = 1 - ssim.mean()
        return loss


# ============================================================================
# PERCEPTUAL LOSS (VGG-BASED)
# ============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Compares high-level features rather than pixel values.
    Loss = L2 distance in VGG feature space
    """
    
    def __init__(self, layer: str = "relu5_1"):
        """
        Initialize perceptual loss.
        
        Args:
            layer: VGG layer to extract features from
                  Options: "relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"
        """
        super().__init__()
        self.layer = layer
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True)
        
        # Determine which layers to use
        layer_dict = {
            "relu1_1": 2,
            "relu2_1": 7,
            "relu3_1": 12,
            "relu4_1": 21,
            "relu5_1": 30,
        }
        
        max_layer = layer_dict[layer]
        
        # Extract feature extractor
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Normalization constants for ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image to ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            Perceptual loss
        """
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        
        pred_features = self.features(pred_norm)
        target_features = self.features(target_norm)
        
        # L2 distance in feature space
        loss = F.mse_loss(pred_features, target_features)
        
        return loss


# ============================================================================
# EDGE PRESERVATION LOSS
# ============================================================================

class EdgePreservationLoss(nn.Module):
    """
    Edge preservation loss using Sobel filters.
    
    Ensures edges and fine details are preserved.
    Loss = L1 distance of edge maps
    """
    
    def __init__(self):
        """Initialize edge preservation loss."""
        super().__init__()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)
        
        # Create filters for 3 channels
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel filters."""
        edges_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edges_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        return edges
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge preservation loss.
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            Edge preservation loss
        """
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        # L1 loss on edges
        loss = torch.mean(torch.abs(pred_edges - target_edges))
        
        return loss


# ============================================================================
# DIFFUSION CONSISTENCY LOSS
# ============================================================================

class DiffusionConsistencyLoss(nn.Module):
    """
    Diffusion consistency loss.
    
    Enforces that the denoising trajectory is meaningful and consistent.
    Loss = L2 between predicted and actual noise during denoising
    """
    
    def __init__(self):
        """Initialize diffusion consistency loss."""
        super().__init__()
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        actual_noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion consistency loss.
        
        Args:
            predicted_noise: Model's noise prediction
            actual_noise: Ground truth noise
            timesteps: Current timesteps (to weight loss)
        
        Returns:
            Diffusion consistency loss
        """
        # Basic L2 loss
        loss = F.mse_loss(predicted_noise, actual_noise, reduction="none")
        
        # Weight by timestep (early steps more important)
        # Normalize timesteps to [0, 1]
        max_t = timesteps.max().float()
        weights = 1.0 - (timesteps.float() / (max_t + 1e-8))  # Earlier steps get higher weight
        weights = weights.view(-1, 1, 1, 1)
        
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss


# ============================================================================
# HYBRID LOSS COMBINER
# ============================================================================

class HybridLoss(nn.Module):
    """
    Hybrid loss combining all 6 loss components.
    
    Total Loss = Σ(weight_i * loss_i)
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize hybrid loss.
        
        Args:
            config: TrainingConfig instance with loss weights
        """
        super().__init__()
        self.config = config
        self.weights = config.loss_weights
        
        # Initialize all loss components
        self.l2_loss = L2Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss(layer="relu5_1")
        self.edge_loss = EdgePreservationLoss()
        self.diffusion_loss = DiffusionConsistencyLoss()
        
        # Print weights
        self._print_weights()
    
    def _print_weights(self):
        """Print loss weights for debugging."""
        total_weight = sum(self.weights.values())
        print("\n" + "="*70)
        print("⚖️  HYBRID LOSS FUNCTION WEIGHTS")
        print("="*70)
        for name, weight in self.weights.items():
            normalized = weight / total_weight
            print(f"  {name:.<30} {weight:.4f} (norm: {normalized:.4f})")
        print(f"  {'Total Weight':.<30} {total_weight:.4f}")
        print("="*70 + "\n")
    
    def forward(
        self,
        stego: torch.Tensor,
        cover: torch.Tensor,
        recovered_secret: torch.Tensor,
        original_secret: torch.Tensor,
        predicted_noise: Optional[torch.Tensor] = None,
        actual_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hybrid loss.
        
        Args:
            stego: Stego image (batch_size, 3, H, W)
            cover: Cover image (batch_size, 3, H, W)
            recovered_secret: Recovered secret image
            original_secret: Original secret image
            predicted_noise: Predicted noise (optional)
            actual_noise: Actual noise (optional)
            timesteps: Timesteps for diffusion (optional)
        
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        
        # 1. Cover reconstruction loss
        loss_cover = self.l2_loss(stego, cover)
        losses["cover_reconstruction"] = loss_cover
        
        # 2. Secret reconstruction loss
        loss_secret = self.l2_loss(recovered_secret, original_secret)
        losses["secret_reconstruction"] = loss_secret
        
        # 3. SSIM loss
        loss_ssim = self.ssim_loss(stego, cover)
        losses["ssim"] = loss_ssim
        
        # 4. Perceptual loss
        loss_perceptual = self.perceptual_loss(stego, cover)
        losses["perceptual"] = loss_perceptual
        
        # 5. Diffusion consistency loss (optional)
        if predicted_noise is not None and actual_noise is not None and timesteps is not None:
            loss_diffusion = self.diffusion_loss(predicted_noise, actual_noise, timesteps)
            losses["diffusion_consistency"] = loss_diffusion
        else:
            losses["diffusion_consistency"] = torch.tensor(0.0, device=stego.device)
        
        # 6. Edge preservation loss
        loss_edge = self.edge_loss(stego, cover)
        losses["edge_preservation"] = loss_edge
        
        # Compute weighted total loss
        total_loss = (
            self.weights["cover_reconstruction"] * losses["cover_reconstruction"] +
            self.weights["secret_reconstruction"] * losses["secret_reconstruction"] +
            self.weights["ssim"] * losses["ssim"] +
            self.weights["perceptual"] * losses["perceptual"] +
            self.weights["diffusion_consistency"] * losses["diffusion_consistency"] +
            self.weights["edge_preservation"] * losses["edge_preservation"]
        )
        
        return total_loss, losses
    
    def get_loss_summary(self, losses: Dict[str, torch.Tensor]) -> str:
        """Get formatted loss summary."""
        summary = "Loss Components:\n"
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            summary += f"  {name:.<30} {value:.6f}\n"
        return summary


# ============================================================================
# LOSS TRACKING
# ============================================================================

class LossTracker:
    """Track and accumulate losses during training."""
    
    def __init__(self):
        """Initialize loss tracker."""
        self.losses = {}
        self.counts = {}
    
    def update(self, loss_dict: Dict[str, torch.Tensor], batch_size: int = 1):
        """
        Update loss tracker.
        
        Args:
            loss_dict: Dictionary of loss values
            batch_size: Batch size for accumulation
        """
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if name not in self.losses:
                self.losses[name] = 0.0
                self.counts[name] = 0
            
            self.losses[name] += value * batch_size
            self.counts[name] += batch_size
    
    def get_average(self) -> Dict[str, float]:
        """Get average losses."""
        averages = {}
        for name in self.losses:
            if self.counts[name] > 0:
                averages[name] = self.losses[name] / self.counts[name]
            else:
                averages[name] = 0.0
        return averages
    
    def reset(self):
        """Reset tracker."""
        self.losses.clear()
        self.counts.clear()
    
    def summary(self) -> str:
        """Get summary string."""
        averages = self.get_average()
        summary = "Average Losses:\n"
        for name, value in averages.items():
            summary += f"  {name:.<30} {value:.6f}\n"
        return summary


# ============================================================================
# TESTING
# ============================================================================

def test_losses():
    """Test all loss functions."""
    print("\n" + "="*70)
    print("🧪 LOSS FUNCTIONS TESTING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    # Create sample tensors
    cover = torch.randn(batch_size, 3, 256, 256).to(device)
    stego = cover + 0.01 * torch.randn_like(cover)
    secret = torch.randn(batch_size, 3, 256, 256).to(device)
    recovered = secret + 0.02 * torch.randn_like(secret)
    
    print(f"\n✓ Created sample tensors")
    print(f"  Cover shape: {cover.shape}")
    print(f"  Stego shape: {stego.shape}")
    print(f"  Secret shape: {secret.shape}")
    print(f"  Recovered shape: {recovered.shape}")
    
    # Test individual losses
    print(f"\n📊 Testing individual loss functions...")
    
    l2 = L2Loss()
    l2_value = l2(stego, cover)
    print(f"  ✓ L2 Loss: {l2_value.item():.6f}")
    
    print(f"\n  ⏳ Computing SSIM Loss (this takes a moment)...")
    ssim = SSIMLoss()
    ssim_value = ssim(stego, cover)
    print(f"  ✓ SSIM Loss: {ssim_value.item():.6f}")
    
    print(f"\n  ⏳ Computing Perceptual Loss (downloading VGG19)...")
    perceptual = PerceptualLoss()
    perceptual_value = perceptual(stego, cover)
    print(f"  ✓ Perceptual Loss: {perceptual_value.item():.6f}")
    
    print(f"\n  ✓ Computing Edge Preservation Loss...")
    edge = EdgePreservationLoss()
    edge_value = edge(stego, cover)
    print(f"  ✓ Edge Loss: {edge_value.item():.6f}")
    
    # Test hybrid loss
    print(f"\n🔀 Testing hybrid loss...")
    config = DEFAULT_CONFIG.training
    hybrid = HybridLoss(config)
    
    total_loss, loss_dict = hybrid(
        stego, cover, recovered, secret
    )
    
    print(f"  ✓ Total Loss: {total_loss.item():.6f}")
    print(hybrid.get_loss_summary(loss_dict))
    
    # Test loss tracker
    print(f"\n📈 Testing loss tracker...")
    tracker = LossTracker()
    
    for _ in range(5):
        tracker.update(loss_dict, batch_size=batch_size)
    
    print(tracker.summary())
    
    print("="*70)
    print("✅ All loss function tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_losses()
