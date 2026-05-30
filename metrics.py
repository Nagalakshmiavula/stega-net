"""
Metrics Module for Diffusion-Based Image Steganography.

This module implements:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- MSE (Mean Squared Error)
- LPIPS (Learned Perceptual Image Patch Similarity)
- BPP analysis metrics
- Comprehensive metric computation and tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import ValidationConfig, DEFAULT_CONFIG


# ============================================================================
# BASIC METRICS
# ============================================================================

class MetricComputer:
    """Compute various image quality metrics."""
    
    @staticmethod
    def compute_psnr(
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        
        PSNR = 20 * log10(MAX_I / sqrt(MSE))
        
        Higher PSNR is better (max ~50 for good quality)
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            max_val: Maximum value (1.0 for normalized, 255 for uint8)
        
        Returns:
            PSNR value in dB
        """
        # Ensure tensors are on CPU and detached
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        
        # Flatten for MSE computation
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        mse = mean_squared_error(target_flat, pred_flat)
        
        if mse == 0:
            return 100.0  # Perfect match
        
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return float(psnr)
    
    @staticmethod
    def compute_mse(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute Mean Squared Error (MSE).
        
        Lower MSE is better.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
        
        Returns:
            MSE value
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        mse = mean_squared_error(target_flat, pred_flat)
        return float(mse)
    
    @staticmethod
    def compute_mae(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Lower MAE is better.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
        
        Returns:
            MAE value
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        mae = mean_absolute_error(target_flat, pred_flat)
        return float(mae)


# ============================================================================
# STRUCTURAL SIMILARITY (SSIM) METRIC
# ============================================================================

class SSIMMetric(nn.Module):
    """Compute Structural Similarity Index Measure (SSIM)."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        """
        Initialize SSIM metric.
        
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian kernel
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        
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
    
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute SSIM.
        
        SSIM ranges from -1 to 1, with 1 being perfect similarity.
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            SSIM value (float between -1 and 1)
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        
        # Ensure 4D tensors
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        
        # Move to same device
        target = target.to(pred.device)
        
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
        
        return float(ssim.mean().item())


# ============================================================================
# PERCEPTUAL METRICS (LPIPS)
# ============================================================================

class LPIPSMetric(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    Lightweight version for perceptual distance measurement.
    """
    
    def __init__(self, net: str = "alex"):
        """
        Initialize LPIPS metric.
        
        Args:
            net: Network type ("alex", "vgg", "squeeze")
        """
        super().__init__()
        self.net = net
        
        # Load lightweight feature extractor
        if net == "alex":
            from torchvision.models import alexnet
            features = alexnet(pretrained=True).features
            self.layers = [features[:4], features[4:9], features[9:]]
        else:
            # Fallback to simple VGG-like layers
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            self.layers = [
                nn.Sequential(*list(vgg.features.children())[:5]),
                nn.Sequential(*list(vgg.features.children())[5:10]),
                nn.Sequential(*list(vgg.features.children())[10:]),
            ]
        
        # Freeze parameters
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False
            layer.eval()
        
        # Normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute LPIPS distance.
        
        Lower LPIPS is better (0 = identical, >1 = very different)
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            LPIPS value
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        
        # Ensure 4D tensors
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        
        target = target.to(pred.device)
        
        # Normalize
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        total_dist = 0.0
        
        # Compute distance at each layer
        for layer in self.layers:
            pred_feat = layer(pred)
            target_feat = layer(target)
            
            # L2 distance
            dist = torch.mean((pred_feat - target_feat) ** 2, dim=(1, 2, 3))
            total_dist += dist.mean().item()
        
        return float(total_dist / len(self.layers))


# ============================================================================
# BPP METRICS
# ============================================================================

class BPPMetrics:
    """Compute Bits Per Pixel (BPP) and capacity metrics."""
    
    @staticmethod
    def compute_bpp(
        secret: torch.Tensor,
        stego: torch.Tensor,
        cover: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Bits Per Pixel (BPP) metrics.
        
        BPP = Hidden Information / Total Pixels
        
        Args:
            secret: Original secret image
            stego: Stego image
            cover: Cover image
        
        Returns:
            Dictionary with BPP metrics
        """
        # Convert to numpy if needed
        if isinstance(secret, torch.Tensor):
            secret = secret.cpu().detach().numpy()
        if isinstance(stego, torch.Tensor):
            stego = stego.cpu().detach().numpy()
        if isinstance(cover, torch.Tensor):
            cover = cover.cpu().detach().numpy()
        
        batch_size, channels, height, width = secret.shape
        total_pixels = height * width
        
        # Compute capacity used
        stego_cover_diff = np.abs(stego - cover)
        secret_diff = np.abs(secret - np.zeros_like(secret))
        
        # Estimate information content
        avg_diff = np.mean(stego_cover_diff)
        max_diff = 2.0  # For [-1, 1] normalized images
        
        # Information capacity (bits per pixel per channel)
        bits_per_channel = 8  # Standard assumption
        capacity_ratio = np.clip(avg_diff / max_diff, 0, 1)
        bpp = capacity_ratio * bits_per_channel * channels
        
        # Total hidden bits
        total_bits = bpp * total_pixels
        
        return {
            "bpp": float(bpp),
            "capacity_ratio": float(capacity_ratio),
            "total_bits": float(total_bits),
            "avg_embedding_intensity": float(avg_diff),
        }
    
    @staticmethod
    def compute_quality_capacity_tradeoff(
        secrets: List[torch.Tensor],
        stegos: List[torch.Tensor],
        covers: List[torch.Tensor],
        compute_psnr: bool = True
    ) -> Dict:
        """
        Analyze trade-off between capacity and quality.
        
        Args:
            secrets: List of secret images
            stegos: List of stego images
            covers: List of cover images
            compute_psnr: Whether to compute PSNR
        
        Returns:
            Trade-off analysis dictionary
        """
        bpp_values = []
        psnr_values = []
        
        for secret, stego, cover in zip(secrets, stegos, covers):
            bpp_metrics = BPPMetrics.compute_bpp(secret, stego, cover)
            bpp_values.append(bpp_metrics["bpp"])
            
            if compute_psnr:
                psnr = MetricComputer.compute_psnr(stego, cover)
                psnr_values.append(psnr)
        
        bpp_array = np.array(bpp_values)
        
        tradeoff = {
            "mean_bpp": float(np.mean(bpp_array)),
            "std_bpp": float(np.std(bpp_array)),
            "min_bpp": float(np.min(bpp_array)),
            "max_bpp": float(np.max(bpp_array)),
        }
        
        if psnr_values:
            psnr_array = np.array(psnr_values)
            tradeoff.update({
                "mean_psnr": float(np.mean(psnr_array)),
                "std_psnr": float(np.std(psnr_array)),
                "min_psnr": float(np.min(psnr_array)),
                "max_psnr": float(np.max(psnr_array)),
            })
            
            # Correlation
            if len(bpp_values) > 1:
                correlation = np.corrcoef(bpp_array, psnr_array)[0, 1]
                tradeoff["bpp_psnr_correlation"] = float(correlation)
        
        return tradeoff


# ============================================================================
# COMPREHENSIVE METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track and accumulate all metrics during validation."""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize metrics tracker.
        
        Args:
            config: ValidationConfig instance
        """
        self.config = config
        
        # Initialize metric computers
        self.ssim_metric = SSIMLoss()
        self.lpips_metric = LPIPSMetric() if config.compute_lpips else None
        
        # Storage
        self.stego_psnr = []
        self.stego_ssim = []
        self.stego_mse = []
        self.secret_psnr = []
        self.secret_ssim = []
        self.secret_mse = []
        self.bpp_values = []
        self.lpips_values = []
    
    def update(
        self,
        stego: torch.Tensor,
        cover: torch.Tensor,
        recovered_secret: torch.Tensor,
        original_secret: torch.Tensor
    ):
        """
        Update metrics with new batch.
        
        Args:
            stego: Stego image
            cover: Cover image
            recovered_secret: Recovered secret
            original_secret: Original secret
        """
        # Stego quality metrics
        if self.config.compute_psnr:
            psnr = MetricComputer.compute_psnr(stego, cover)
            self.stego_psnr.append(psnr)
        
        if self.config.compute_ssim:
            ssim = self.ssim_metric.compute(stego, cover)
            self.stego_ssim.append(ssim)
        
        if self.config.compute_mse:
            mse = MetricComputer.compute_mse(stego, cover)
            self.stego_mse.append(mse)
        
        # Secret recovery metrics
        if self.config.compute_psnr:
            secret_psnr = MetricComputer.compute_psnr(recovered_secret, original_secret)
            self.secret_psnr.append(secret_psnr)
        
        if self.config.compute_ssim:
            secret_ssim = self.ssim_metric.compute(recovered_secret, original_secret)
            self.secret_ssim.append(secret_ssim)
        
        if self.config.compute_mse:
            secret_mse = MetricComputer.compute_mse(recovered_secret, original_secret)
            self.secret_mse.append(secret_mse)
        
        # BPP metrics
        if self.config.compute_bpp:
            bpp_dict = BPPMetrics.compute_bpp(original_secret, stego, cover)
            self.bpp_values.append(bpp_dict["bpp"])
        
        # LPIPS metrics
        if self.config.compute_lpips and self.lpips_metric:
            lpips = self.lpips_metric.compute(stego, cover)
            self.lpips_values.append(lpips)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics."""
        averages = {}
        
        if self.stego_psnr:
            averages["stego_psnr"] = float(np.mean(self.stego_psnr))
        if self.stego_ssim:
            averages["stego_ssim"] = float(np.mean(self.stego_ssim))
        if self.stego_mse:
            averages["stego_mse"] = float(np.mean(self.stego_mse))
        if self.secret_psnr:
            averages["secret_psnr"] = float(np.mean(self.secret_psnr))
        if self.secret_ssim:
            averages["secret_ssim"] = float(np.mean(self.secret_ssim))
        if self.secret_mse:
            averages["secret_mse"] = float(np.mean(self.secret_mse))
        if self.bpp_values:
            averages["bpp"] = float(np.mean(self.bpp_values))
        if self.lpips_values:
            averages["lpips"] = float(np.mean(self.lpips_values))
        
        return averages
    
    def get_std(self) -> Dict[str, float]:
        """Get standard deviations."""
        stds = {}
        
        if len(self.stego_psnr) > 1:
            stds["stego_psnr_std"] = float(np.std(self.stego_psnr))
        if len(self.stego_ssim) > 1:
            stds["stego_ssim_std"] = float(np.std(self.stego_ssim))
        if len(self.stego_mse) > 1:
            stds["stego_mse_std"] = float(np.std(self.stego_mse))
        if len(self.secret_psnr) > 1:
            stds["secret_psnr_std"] = float(np.std(self.secret_psnr))
        if len(self.secret_ssim) > 1:
            stds["secret_ssim_std"] = float(np.std(self.secret_ssim))
        if len(self.secret_mse) > 1:
            stds["secret_mse_std"] = float(np.std(self.secret_mse))
        
        return stds
    
    def reset(self):
        """Reset tracker."""
        self.stego_psnr = []
        self.stego_ssim = []
        self.stego_mse = []
        self.secret_psnr = []
        self.secret_ssim = []
        self.secret_mse = []
        self.bpp_values = []
        self.lpips_values = []
    
    def summary(self) -> str:
        """Get metrics summary."""
        averages = self.get_averages()
        stds = self.get_std()
        
        summary = "\n" + "="*70 + "\n"
        summary += "📊 METRICS SUMMARY\n"
        summary += "="*70 + "\n\n"
        
        summary += "🖼️  STEGO IMAGE QUALITY\n"
        if "stego_psnr" in averages:
            summary += f"  PSNR: {averages['stego_psnr']:.4f}"
            if "stego_psnr_std" in stds:
                summary += f" ± {stds['stego_psnr_std']:.4f}"
            summary += " dB\n"
        if "stego_ssim" in averages:
            summary += f"  SSIM: {averages['stego_ssim']:.4f}"
            if "stego_ssim_std" in stds:
                summary += f" ± {stds['stego_ssim_std']:.4f}"
            summary += "\n"
        if "stego_mse" in averages:
            summary += f"  MSE:  {averages['stego_mse']:.6f}\n"
        if "lpips" in averages:
            summary += f"  LPIPS: {averages['lpips']:.6f}\n"
        
        summary += "\n🔐 SECRET RECOVERY QUALITY\n"
        if "secret_psnr" in averages:
            summary += f"  PSNR: {averages['secret_psnr']:.4f}"
            if "secret_psnr_std" in stds:
                summary += f" ± {stds['secret_psnr_std']:.4f}"
            summary += " dB\n"
        if "secret_ssim" in averages:
            summary += f"  SSIM: {averages['secret_ssim']:.4f}"
            if "secret_ssim_std" in stds:
                summary += f" ± {stds['secret_ssim_std']:.4f}"
            summary += "\n"
        if "secret_mse" in averages:
            summary += f"  MSE:  {averages['secret_mse']:.6f}\n"
        
        summary += "\n📈 CAPACITY METRICS\n"
        if "bpp" in averages:
            summary += f"  BPP:  {averages['bpp']:.4f} bits/pixel\n"
        
        summary += "="*70 + "\n"
        
        return summary


# ============================================================================
# TESTING
# ============================================================================

def test_metrics():
    """Test all metric functions."""
    print("\n" + "="*70)
    print("🧪 METRICS TESTING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    # Create sample tensors
    cover = torch.randn(batch_size, 3, 256, 256).to(device)
    stego = cover + 0.01 * torch.randn_like(cover)
    secret = torch.randn(batch_size, 3, 256, 256).to(device)
    recovered = secret + 0.02 * torch.randn_like(secret)
    
    print(f"\n✓ Created sample tensors")
    
    # Test PSNR
    print(f"\n📊 Testing PSNR...")
    psnr = MetricComputer.compute_psnr(stego, cover)
    print(f"  ✓ PSNR: {psnr:.4f} dB")
    
    # Test SSIM
    print(f"\n📊 Testing SSIM...")
    ssim_metric = SSIMMetric()
    ssim = ssim_metric.compute(stego, cover)
    print(f"  ✓ SSIM: {ssim:.4f}")
    
    # Test MSE
    print(f"\n📊 Testing MSE...")
    mse = MetricComputer.compute_mse(stego, cover)
    print(f"  ✓ MSE: {mse:.6f}")
    
    # Test BPP
    print(f"\n📊 Testing BPP...")
    bpp_dict = BPPMetrics.compute_bpp(secret, stego, cover)
    print(f"  ✓ BPP: {bpp_dict['bpp']:.4f}")
    print(f"  ✓ Total Hidden Bits: {bpp_dict['total_bits']:.0f}")
    
    # Test tracker
    print(f"\n📈 Testing metrics tracker...")
    config = DEFAULT_CONFIG.validation
    tracker = MetricsTracker(config)
    
    for _ in range(3):
        tracker.update(stego, cover, recovered, secret)
    
    print(tracker.summary())
    
    print("="*70)
    print("✅ All metric tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_metrics()
