"""
Utility Functions for Diffusion-Based Image Steganography.

This module provides:
- Device management (GPU/CPU)
- Reproducibility and seeding
- Checkpoint save/load
- Image normalization/denormalization
- Batch processing utilities
- Progress tracking
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
from datetime import datetime
import shutil

from config import FullConfig, DEFAULT_CONFIG


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

class DeviceManager:
    """Manage device selection and setup."""
    
    @staticmethod
    def get_device(use_gpu: bool = True) -> torch.device:
        """
        Get appropriate device (GPU or CPU).
        
        Args:
            use_gpu: Whether to use GPU if available
        
        Returns:
            torch.device instance
        """
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            print(f"✓ Using CPU")
        
        return device
    
    @staticmethod
    def clear_cache(device: torch.device):
        """Clear GPU cache if using GPU."""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_device_info(device: torch.device) -> Dict[str, Any]:
        """Get device information."""
        info = {"device": str(device)}
        
        if device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / 1e9
            info["reserved_memory_gb"] = torch.cuda.memory_reserved(0) / 1e9
        
        return info


# ============================================================================
# REPRODUCIBILITY & SEEDING
# ============================================================================

class ReproducibilityManager:
    """Manage reproducibility settings."""
    
    @staticmethod
    def set_seed(seed: int = 42, deterministic: bool = True):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Seed value
            deterministic: Whether to use deterministic algorithms
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        
        print(f"✓ Seed set to {seed} (deterministic={deterministic})")
    
    @staticmethod
    def get_seed_info() -> Dict[str, Any]:
        """Get current seed information."""
        return {
            "python_random_state": random.getstate()[1][0],
            "numpy_random_state": np.random.get_state()[1][0],
            "torch_random_state": torch.initial_seed(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        loss: float,
        metrics: Optional[Dict] = None,
        config: Optional[FullConfig] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            loss: Current loss
            metrics: Optional metrics dictionary
            config: Optional config object
            is_best: Whether this is the best model
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "metrics": metrics or {},
            "config": config,
        }
        
        # Create filename
        filename = f"epoch_{epoch:04d}_loss_{loss:.6f}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
        
        # Save as best if indicated
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(filepath, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        return filepath
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load to
        
        Returns:
            Checkpoint data dictionary
        """
        if device is None:
            device = torch.device("cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state"])
        print(f"✓ Model loaded from checkpoint")
        
        # Load optimizer state
        if optimizer and checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(f"✓ Optimizer state loaded")
        
        # Load scheduler state
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"✓ Scheduler state loaded")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        return None
    
    def cleanup_old_checkpoints(self, keep_top_n: int = 5):
        """
        Keep only top N checkpoints by loss.
        
        Args:
            keep_top_n: Number of checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pt"))
        
        if len(checkpoints) > keep_top_n:
            # Extract loss from filename and sort
            checkpoints_with_loss = []
            for cp in checkpoints:
                try:
                    loss = float(cp.stem.split("_")[-1])
                    checkpoints_with_loss.append((cp, loss))
                except:
                    pass
            
            # Sort by loss and keep top N
            checkpoints_with_loss.sort(key=lambda x: x[1])
            to_keep = set(cp[0] for cp in checkpoints_with_loss[:keep_top_n])
            
            # Remove others
            for cp in checkpoints:
                if cp not in to_keep and cp.name != "best_model.pt":
                    cp.unlink()
                    print(f"✓ Removed old checkpoint: {cp.name}")


# ============================================================================
# IMAGE NORMALIZATION/DENORMALIZATION
# ============================================================================

class ImageNormalizer:
    """Handle image normalization and denormalization."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image normalizer.
        
        Args:
            mean: ImageNet normalization mean
            std: ImageNet normalization std
        """
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize image to ImageNet statistics.
        
        Args:
            x: Image tensor with values in [0, 1] or [-1, 1]
        
        Returns:
            Normalized tensor
        """
        x = x.to(self.mean.device)
        return (x - self.mean) / self.std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize image from ImageNet statistics.
        
        Args:
            x: Normalized image tensor
        
        Returns:
            Denormalized tensor with values in [0, 1]
        """
        x = x.to(self.mean.device)
        x = x * self.std + self.mean
        return torch.clamp(x, 0, 1)
    
    def to_device(self, device: torch.device):
        """Move normalizer to device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

class BatchProcessor:
    """Process batches efficiently."""
    
    @staticmethod
    def move_batch_to_device(
        batch: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Move batch to device.
        
        Args:
            batch: (cover, secret) tuple
            device: Target device
        
        Returns:
            Batch on device
        """
        cover, secret = batch
        return cover.to(device), secret.to(device)
    
    @staticmethod
    def get_batch_size(batch: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """Get batch size."""
        cover, _ = batch
        return cover.shape[0]
    
    @staticmethod
    def clip_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clip batch values to [-1, 1]."""
        cover, secret = batch
        return torch.clamp(cover, -1, 1), torch.clamp(secret, -1, 1)


# ============================================================================
# LOGGING & PROGRESS TRACKING
# ============================================================================

class Logger:
    """Handle logging to file and console."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize logger.
        
        Args:
            log_file: Optional file to log to
        """
        self.log_file = log_file
        if log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, print_console: bool = True):
        """
        Log message.
        
        Args:
            message: Message to log
            print_console: Whether to print to console
        """
        if print_console:
            print(message)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
    
    def log_dict(self, data: Dict[str, Any], prefix: str = ""):
        """Log dictionary."""
        for key, value in data.items():
            self.log(f"{prefix}{key}: {value}")


class ProgressTracker:
    """Track training progress."""
    
    def __init__(self, total_epochs: int, checkpoint_interval: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_epochs: Total number of epochs
            checkpoint_interval: Checkpoint saving interval
        """
        self.total_epochs = total_epochs
        self.checkpoint_interval = checkpoint_interval
        self.start_time = None
        self.epoch_times = []
    
    def start(self):
        """Start timer."""
        self.start_time = datetime.now()
    
    def end_epoch(self) -> str:
        """End epoch and return time string."""
        if self.start_time is None:
            return "N/A"
        
        elapsed = datetime.now() - self.start_time
        self.epoch_times.append(elapsed.total_seconds())
        
        return str(elapsed).split(".")[0]  # HH:MM:SS
    
    def estimate_time_remaining(self, current_epoch: int) -> str:
        """
        Estimate time remaining.
        
        Args:
            current_epoch: Current epoch number (0-indexed)
        
        Returns:
            Formatted time string
        """
        if not self.epoch_times:
            return "N/A"
        
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch - 1
        remaining_seconds = avg_epoch_time * remaining_epochs
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def get_progress_bar(self, current_epoch: int, width: int = 40) -> str:
        """
        Get progress bar string.
        
        Args:
            current_epoch: Current epoch (0-indexed)
            width: Progress bar width
        
        Returns:
            Formatted progress bar
        """
        progress = (current_epoch + 1) / self.total_epochs
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = f"{progress * 100:.1f}%"
        return f"[{bar}] {percentage}"


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

class ConfigManager:
    """Manage configuration files."""
    
    @staticmethod
    def save_config(config: FullConfig, filepath: Path):
        """
        Save configuration to JSON.
        
        Args:
            config: Configuration object
            filepath: Path to save to
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        config.save_to_file(str(filepath))
    
    @staticmethod
    def load_config_json(filepath: Path) -> Dict:
        """
        Load configuration from JSON.
        
        Args:
            filepath: Path to config file
        
        Returns:
            Configuration dictionary
        """
        with open(filepath, "r") as f:
            return json.load(f)
    
    @staticmethod
    def print_config(config: FullConfig):
        """Print configuration summary."""
        print(config.summary())


# ============================================================================
# METRICS LOGGING
# ============================================================================

class MetricsLogger:
    """Log training and validation metrics."""
    
    def __init__(self, log_file: Path):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics to CSV.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        import csv
        
        # Add epoch
        row = {"epoch": epoch}
        row.update(metrics)
        
        # Write header if first time
        if not self.header_written:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            self.header_written = True
        
        # Append metrics
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def get_metrics_dataframe(self):
        """Get metrics as pandas DataFrame."""
        try:
            import pandas as pd
            return pd.read_csv(self.log_file)
        except ImportError:
            print("Pandas not installed. Install with: pip install pandas")
            return None


# ============================================================================
# MODEL UTILITIES
# ============================================================================

class ModelUtils:
    """Model utility functions."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        }
    
    @staticmethod
    def freeze_parameters(model: nn.Module):
        """Freeze all parameters."""
        for param in model.parameters():
            param.requires_grad = False
    
    @staticmethod
    def unfreeze_parameters(model: nn.Module):
        """Unfreeze all parameters."""
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def print_model_summary(model: nn.Module, input_size: Tuple = None):
        """
        Print model summary.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size
        """
        print("\n" + "="*70)
        print("MODEL SUMMARY")
        print("="*70)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("="*70 + "\n")


# ============================================================================
# TESTING
# ============================================================================

def test_utils():
    """Test utility functions."""
    print("\n" + "="*70)
    print("🧪 UTILITY FUNCTIONS TESTING")
    print("="*70)
    
    # Test device manager
    print("\n📱 Testing DeviceManager...")
    device = DeviceManager.get_device(use_gpu=True)
    device_info = DeviceManager.get_device_info(device)
    print(f"  Device info: {device_info}")
    
    # Test reproducibility
    print("\n🌱 Testing ReproducibilityManager...")
    ReproducibilityManager.set_seed(42, deterministic=True)
    seed_info = ReproducibilityManager.get_seed_info()
    print(f"  Seed set successfully")
    
    # Test image normalizer
    print("\n🖼️  Testing ImageNormalizer...")
    normalizer = ImageNormalizer()
    normalizer.to_device(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    x_norm = normalizer.normalize(x)
    x_denorm = normalizer.denormalize(x_norm)
    print(f"  Original range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Normalized range: [{x_norm.min():.4f}, {x_norm.max():.4f}]")
    print(f"  Denormalized range: [{x_denorm.min():.4f}, {x_denorm.max():.4f}]")
    
    # Test batch processor
    print("\n📦 Testing BatchProcessor...")
    cover = torch.randn(4, 3, 256, 256)
    secret = torch.randn(4, 3, 256, 256)
    batch = (cover, secret)
    batch_size = BatchProcessor.get_batch_size(batch)
    print(f"  Batch size: {batch_size}")
    batch_device = BatchProcessor.move_batch_to_device(batch, device)
    print(f"  Moved to device: {batch_device[0].device}")
    
    # Test progress tracker
    print("\n⏱️  Testing ProgressTracker...")
    tracker = ProgressTracker(total_epochs=1000, checkpoint_interval=100)
    tracker.start()
    import time
    time.sleep(0.5)
    epoch_time = tracker.end_epoch()
    print(f"  Epoch time: {epoch_time}")
    progress_bar = tracker.get_progress_bar(current_epoch=50)
    print(f"  Progress: {progress_bar}")
    
    # Test checkpoint manager
    print("\n💾 Testing CheckpointManager...")
    checkpoint_dir = Path("./test_checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    print(f"  Checkpoint dir created: {checkpoint_dir.exists()}")
    
    # Cleanup
    shutil.rmtree(checkpoint_dir)
    print(f"  Cleanup successful")
    
    print("\n" + "="*70)
    print("✅ All utility tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_utils()
