"""
Diffusion Process Module for Image Steganography.

This module implements:
- Noise schedules (linear, cosine, sqrt)
- Forward diffusion process (add noise)
- Reverse diffusion utilities
- Bits Per Pixel (BPP) calculation
- Noise prediction and sampling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import math

from config import DiffusionConfig, DEFAULT_CONFIG


# ============================================================================
# NOISE SCHEDULES
# ============================================================================

class NoiseSchedule:
    """Base class for noise schedules."""
    
    def __init__(self, config: DiffusionConfig):
        """
        Initialize noise schedule.
        
        Args:
            config: DiffusionConfig instance
        """
        self.config = config
        self.num_steps = config.num_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        
        # Pre-compute schedules
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Pre-compute derived quantities
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)
    
    def _compute_betas(self) -> np.ndarray:
        """Compute beta schedule (to be overridden by subclasses)."""
        raise NotImplementedError
    
    def get_sqrt_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get sqrt(alpha_cumprod) for timestep t.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            Values of shape (batch_size,)
        """
        indices = t.cpu().long().numpy()
        return torch.from_numpy(
            self.sqrt_alphas_cumprod[indices]
        ).to(t.device).float()
    
    def get_sqrt_one_minus_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get sqrt(1 - alpha_cumprod) for timestep t.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            Values of shape (batch_size,)
        """
        indices = t.cpu().long().numpy()
        return torch.from_numpy(
            self.sqrt_one_minus_alphas_cumprod[indices]
        ).to(t.device).float()
    
    def get_sqrt_recip_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get sqrt(1 / alpha_cumprod) for timestep t.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            Values of shape (batch_size,)
        """
        indices = t.cpu().long().numpy()
        return torch.from_numpy(
            self.sqrt_recip_alphas_cumprod[indices]
        ).to(t.device).float()
    
    def get_posterior_variance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get posterior variance for timestep t.
        
        Used in reverse diffusion process.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            Posterior variance values of shape (batch_size,)
        """
        indices = t.cpu().long().numpy()
        
        posterior_variance = (
            self.betas[indices] *
            (1.0 - self.alphas_cumprod_prev[indices]) /
            (1.0 - self.alphas_cumprod[indices])
        )
        
        return torch.from_numpy(posterior_variance).to(t.device).float()
    
    def get_betas(self) -> torch.Tensor:
        """Get all beta values."""
        return torch.from_numpy(self.betas).float()
    
    def summary(self) -> str:
        """Return schedule summary."""
        return f"{self.__class__.__name__}: steps={self.num_steps}, beta=[{self.beta_start:.4f}, {self.beta_end:.4f}]"


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule."""
    
    def _compute_betas(self) -> np.ndarray:
        """
        Linear schedule from beta_start to beta_end.
        
        Returns:
            Beta values of shape (num_steps,)
        """
        return np.linspace(
            self.beta_start,
            self.beta_end,
            self.num_steps,
            dtype=np.float64
        )


class CosineSchedule(NoiseSchedule):
    """Cosine annealing noise schedule."""
    
    def _compute_betas(self) -> np.ndarray:
        """
        Cosine annealing schedule.
        
        Based on "Improved Denoising Diffusion Probabilistic Models"
        
        Returns:
            Beta values of shape (num_steps,)
        """
        s = 0.008
        steps = np.arange(self.num_steps + 1, dtype=np.float64)
        
        alphas_cumprod = np.cos(
            ((steps / self.num_steps) + s) / (1 + s) * np.pi * 0.5
        ) ** 2
        
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, self.beta_start, self.beta_end)
        
        return betas


class SqrtSchedule(NoiseSchedule):
    """Square-root noise schedule."""
    
    def _compute_betas(self) -> np.ndarray:
        """
        Square-root schedule for smoother progression.
        
        Returns:
            Beta values of shape (num_steps,)
        """
        steps = np.arange(self.num_steps, dtype=np.float64)
        normalized_steps = steps / self.num_steps
        
        betas = self.beta_start + (self.beta_end - self.beta_start) * np.sqrt(normalized_steps)
        
        return betas


# ============================================================================
# DIFFUSION SCHEDULER
# ============================================================================

class DiffusionScheduler:
    """Main diffusion scheduler managing noise schedules and processes."""
    
    def __init__(self, config: DiffusionConfig):
        """
        Initialize diffusion scheduler.
        
        Args:
            config: DiffusionConfig instance
        """
        self.config = config
        self.num_steps = config.num_steps
        
        # Select schedule
        if config.schedule_type == "linear":
            self.schedule = LinearSchedule(config)
        elif config.schedule_type == "cosine":
            self.schedule = CosineSchedule(config)
        elif config.schedule_type == "sqrt":
            self.schedule = SqrtSchedule(config)
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
        
        # Move schedules to torch tensors
        self._register_schedules()
    
    def _register_schedules(self):
        """Convert schedules to tensors for efficient computation."""
        self.betas = torch.from_numpy(self.schedule.betas).float()
        self.alphas = torch.from_numpy(self.schedule.alphas).float()
        self.alphas_cumprod = torch.from_numpy(self.schedule.alphas_cumprod).float()
        self.sqrt_alphas_cumprod = torch.from_numpy(self.schedule.sqrt_alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            self.schedule.sqrt_one_minus_alphas_cumprod
        ).float()
    
    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to image (forward diffusion process).
        
        Implements: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps
        
        Args:
            x0: Original image (batch_size, channels, H, W)
            t: Timesteps (batch_size,)
            noise: Predefined noise (optional)
        
        Returns:
            (noisy_image, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        device = x0.device
        self._move_to_device(device)
        
        # Get coefficients for this timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)
        
        # Add noise
        x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps.
        
        Args:
            batch_size: Number of samples
            device: Target device
        
        Returns:
            Timestep tensor of shape (batch_size,)
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device)
    
    def _move_to_device(self, device: torch.device):
        """Move schedules to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
    
    def summary(self) -> str:
        """Return scheduler summary."""
        return f"""
╔════════════════════════════════════════════════╗
║       DIFFUSION SCHEDULER CONFIGURATION        ║
╚════════════════════════════════════════════════╝

📊 Schedule Details
   • Type: {self.config.schedule_type.upper()}
   • Steps: {self.num_steps}
   • Beta Start: {self.config.beta_start}
   • Beta End: {self.config.beta_end}
   • Noise Type: {self.config.noise_type}
   • Predict Type: {self.config.predict_type}

🔄 Injection Schedule
   • Strategy: {self.config.secret_injection_schedule}
   • Strength per Step: {self.config.secret_strength_per_step}

📈 Schedule Values
   • Beta Range: [{self.betas.min():.6f}, {self.betas.max():.6f}]
   • Alpha CumProd Range: [{self.alphas_cumprod.min():.6f}, {self.alphas_cumprod.max():.6f}]
        """


# ============================================================================
# BITS PER PIXEL (BPP) CALCULATION
# ============================================================================

class BitsPerPixelCalculator:
    """Calculate and analyze Bits Per Pixel (payload capacity)."""
    
    @staticmethod
    def calculate_bpp(
        secret_image: torch.Tensor,
        stego_image: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate BPP for embedded secret.
        
        BPP = Hidden Bits / Total Pixels
        
        Args:
            secret_image: Original secret image (batch_size, 3, H, W)
            stego_image: Stego image (batch_size, 3, H, W)
        
        Returns:
            Dictionary with BPP metrics
        """
        batch_size, channels, height, width = secret_image.shape
        total_pixels = height * width
        
        # Calculate difference between stego and secret
        # This gives us embedding capacity
        difference = (stego_image - secret_image).abs()
        
        # Estimate bits based on difference magnitude
        # Higher difference = more embedding capacity used
        avg_difference = difference.mean().item()
        
        # Convert to BPP (simplified: assuming 8-bit per pixel per channel)
        # More sophisticated: use information theory (entropy)
        max_difference = 2.0  # For [-1, 1] normalized images
        bits_per_channel = 8  # Standard assumption
        
        # Capacity: how many bits per pixel can be hidden
        embedding_ratio = (avg_difference / max_difference)
        bpp = embedding_ratio * bits_per_channel * channels
        
        # Calculate PSNR to assess quality vs capacity trade-off
        mse = torch.mean((stego_image - secret_image) ** 2).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        return {
            "bpp": bpp,
            "embedding_ratio": embedding_ratio,
            "avg_difference": avg_difference,
            "psnr": psnr,
            "total_pixels": total_pixels,
            "total_bits_capacity": bpp * total_pixels,
        }
    
    @staticmethod
    def calculate_capacity_quality_tradeoff(
        secret_images: List[torch.Tensor],
        stego_images: List[torch.Tensor],
        quality_thresholds: List[float] = None
    ) -> Dict:
        """
        Analyze trade-off between payload capacity and image quality.
        
        Args:
            secret_images: List of secret images
            stego_images: List of stego images
            quality_thresholds: PSNR thresholds for analysis
        
        Returns:
            Analysis dictionary
        """
        if quality_thresholds is None:
            quality_thresholds = [0.99, 0.95, 0.90]
        
        bpp_values = []
        psnr_values = []
        
        for secret, stego in zip(secret_images, stego_images):
            metrics = BitsPerPixelCalculator.calculate_bpp(secret, stego)
            bpp_values.append(metrics["bpp"])
            psnr_values.append(metrics["psnr"])
        
        bpp_array = np.array(bpp_values)
        psnr_array = np.array(psnr_values)
        
        analysis = {
            "mean_bpp": float(np.mean(bpp_array)),
            "std_bpp": float(np.std(bpp_array)),
            "min_bpp": float(np.min(bpp_array)),
            "max_bpp": float(np.max(bpp_array)),
            "mean_psnr": float(np.mean(psnr_array)),
            "std_psnr": float(np.std(psnr_array)),
            "min_psnr": float(np.min(psnr_array)),
            "max_psnr": float(np.max(psnr_array)),
        }
        
        # Correlation analysis
        if len(bpp_values) > 1:
            correlation = np.corrcoef(bpp_array, psnr_array)[0, 1]
            analysis["bpp_psnr_correlation"] = float(correlation)
        
        return analysis


# ============================================================================
# DIFFUSION UTILITIES
# ============================================================================

class DiffusionUtils:
    """Utility functions for diffusion processes."""
    
    @staticmethod
    def predict_noise_from_x0(
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
        scheduler: DiffusionScheduler
    ) -> torch.Tensor:
        """
        Predict noise from x_t and x_0.
        
        Inverse of add_noise: eps = (x_t - sqrt(alpha_cumprod) * x_0) / sqrt(1 - alpha_cumprod)
        
        Args:
            x_t: Noisy image at time t
            t: Timestep
            x_0: Original image
            scheduler: DiffusionScheduler instance
        
        Returns:
            Predicted noise
        """
        device = x_t.device
        scheduler._move_to_device(device)
        
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)
        
        predicted_noise = (x_t - sqrt_alpha * x_0) / sqrt_one_minus_alpha
        
        return predicted_noise
    
    @staticmethod
    def denoise_step(
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        scheduler: DiffusionScheduler
    ) -> torch.Tensor:
        """
        Single denoising step.
        
        Args:
            model_output: Model prediction (noise or sample)
            timestep: Current timestep
            sample: Current noisy sample
            scheduler: DiffusionScheduler instance
        
        Returns:
            Denoised sample
        """
        device = sample.device
        scheduler._move_to_device(device)
        
        # Get coefficients
        t = timestep.cpu().long().numpy()
        
        prev_timestep = timestep - 1
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[np.maximum(prev_timestep.cpu().numpy(), 0)]
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Prediction type: residual (predict noise)
        pred_original_sample = (
            sample - np.sqrt(beta_prod_t) * model_output
        ) / np.sqrt(alpha_prod_t)
        
        # Clip to [-1, 1]
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Calculate variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = torch.clamp(torch.tensor(variance), min=1e-20)
        
        # Predict previous sample
        pred_prev_sample = (
            np.sqrt(alpha_prod_t_prev) * pred_original_sample +
            np.sqrt(1 - alpha_prod_t_prev - variance) * model_output
        )
        
        return pred_prev_sample
    
    @staticmethod
    def get_noise_schedule_summary(scheduler: DiffusionScheduler) -> str:
        """Get detailed noise schedule summary."""
        betas = scheduler.betas.cpu().numpy()
        alphas = scheduler.alphas.cpu().numpy()
        alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
        
        summary = f"""
╔════════════════════════════════════════════════╗
║      NOISE SCHEDULE DETAILED ANALYSIS          ║
╚════════════════════════════════════════════════╝

📊 BETA VALUES
   • Min: {betas.min():.6f}
   • Max: {betas.max():.6f}
   • Mean: {betas.mean():.6f}
   • Std: {betas.std():.6f}

📈 ALPHA VALUES
   • Min: {alphas.min():.6f}
   • Max: {alphas.max():.6f}
   • Mean: {alphas.mean():.6f}

📊 CUMULATIVE PRODUCT
   • Alpha CumProd[0]: {alphas_cumprod[0]:.6f} (no noise)
   • Alpha CumProd[T-1]: {alphas_cumprod[-1]:.6f} (fully noisy)
   • Decay Rate: {alphas_cumprod[0] - alphas_cumprod[-1]:.6f}

🔍 NOISE PROGRESSION
   Step 0: noise_ratio = {1 - alphas_cumprod[0]:.6f}
   Step {len(betas)//4}: noise_ratio = {1 - alphas_cumprod[len(betas)//4]:.6f}
   Step {len(betas)//2}: noise_ratio = {1 - alphas_cumprod[len(betas)//2]:.6f}
   Step {3*len(betas)//4}: noise_ratio = {1 - alphas_cumprod[3*len(betas)//4]:.6f}
   Step {len(betas)-1}: noise_ratio = {1 - alphas_cumprod[-1]:.6f}
        """
        return summary


# ============================================================================
# TESTING
# ============================================================================

def test_diffusion():
    """Test diffusion scheduler and noise schedules."""
    print("\n" + "="*70)
    print("🧪 DIFFUSION MODULE TESTING")
    print("="*70)
    
    config = DEFAULT_CONFIG.diffusion
    
    # Test all schedule types
    for schedule_type in ["linear", "cosine", "sqrt"]:
        print(f"\n📊 Testing {schedule_type.upper()} schedule...")
        config.schedule_type = schedule_type
        scheduler = DiffusionScheduler(config)
        print(scheduler.summary())
    
    # Test noise addition
    print("\n🔄 Testing noise addition...")
    config.schedule_type = "cosine"
    scheduler = DiffusionScheduler(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    x0 = torch.randn(batch_size, 3, 256, 256).to(device)
    t = torch.tensor([0, config.num_steps - 1]).to(device)
    
    x_t, noise = scheduler.add_noise(x0, t)
    print(f"   ✓ Original shape: {x0.shape}")
    print(f"   ✓ Noisy shape: {x_t.shape}")
    print(f"   ✓ Noise shape: {noise.shape}")
    
    # Test BPP calculation
    print("\n📈 Testing BPP calculation...")
    secret = torch.randn(batch_size, 3, 256, 256)
    stego = secret + 0.05 * torch.randn_like(secret)
    
    bpp_metrics = BitsPerPixelCalculator.calculate_bpp(secret, stego)
    print(f"   ✓ BPP: {bpp_metrics['bpp']:.4f}")
    print(f"   ✓ PSNR: {bpp_metrics['psnr']:.2f}")
    print(f"   ✓ Total Capacity: {bpp_metrics['total_bits_capacity']:.0f} bits")
    
    print("\n" + "="*70)
    print("✅ All diffusion tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_diffusion()
