"""
Main Training Loop for Diffusion-Based Image Steganography.

This module implements:
- Complete training pipeline (1000 epochs)
- Loss computation with hybrid loss function
- Optimization and backward pass
- Checkpoint saving every 100 epochs
- Validation every 100 epochs
- Metrics tracking and convergence analysis
- Sample visualization
- Early stopping capability
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
from datetime import datetime

from config import DEFAULT_CONFIG, DatasetConfig, FullConfig
from dataset import create_dataloaders
from model import DiffusionSteganography
from diffusion import DiffusionScheduler
from losses import HybridLoss, LossTracker
from metrics import MetricsTracker
from utils import (
    DeviceManager, ReproducibilityManager, CheckpointManager,
    ProgressTracker, Logger, MetricsLogger, ModelUtils,
    ConfigManager, BatchProcessor
)


# ============================================================================
# TRAINER CLASS
# ============================================================================

class DiffusionSteganographyTrainer:
    """Complete trainer for diffusion-based steganography."""
    
    def __init__(
        self,
        config: FullConfig,
        device: torch.device,
        logger: Logger
    ):
        """
        Initialize trainer.
        
        Args:
            config: Full configuration
            device: Device to train on
            logger: Logger instance
        """
        self.config = config
        self.device = device
        self.logger = logger
        
        # Initialize components
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_loss()
        self._initialize_managers()
    
    def _initialize_model(self):
        """Initialize model."""
        self.logger.log("\n📦 Initializing Model...")
        
        self.model = DiffusionSteganography(
            model_config=self.config.model,
            diffusion_config=self.config.diffusion
        ).to(self.device)
        
        self.model.print_architecture()
        
        param_counts = ModelUtils.count_parameters(self.model)
        self.logger.log(
            f"✓ Model initialized\n"
            f"  Total params: {param_counts['total']:,}\n"
            f"  Trainable params: {param_counts['trainable']:,}\n"
        )
    
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        self.logger.log("\n⚙️  Initializing Optimizer...")
        
        optimizer_type = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.logger.log(f"✓ {optimizer_type.upper()} optimizer initialized\n")
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler."""
        self.logger.log("📊 Initializing LR Scheduler...")
        
        scheduler_type = self.config.training.scheduler_type.lower()
        warmup_epochs = self.config.training.warmup_epochs
        min_lr = self.config.training.min_learning_rate
        
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=min_lr
            )
        elif scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_lr / self.config.training.learning_rate,
                total_iters=self.config.training.num_epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        
        self.logger.log(f"✓ {scheduler_type.upper()} scheduler initialized\n")
    
    def _initialize_loss(self):
        """Initialize loss function."""
        self.logger.log("📌 Initializing Loss Function...")
        self.loss_fn = HybridLoss(self.config.training)
    
    def _initialize_managers(self):
        """Initialize checkpoint and metrics managers."""
        self.checkpoint_manager = CheckpointManager(self.config.checkpoints_dir)
        self.metrics_logger = MetricsLogger(self.config.logs_dir / "metrics.csv")
        self.progress_tracker = ProgressTracker(
            self.config.training.num_epochs,
            self.config.training.checkpoint_interval
        )
    
    def train_epoch(
        self,
        train_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            epoch: Current epoch number (0-indexed)
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        loss_tracker = LossTracker()
        
        self.progress_tracker.start()
        
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            cover, secret = BatchProcessor.move_batch_to_device(batch, self.device)
            batch_size = BatchProcessor.get_batch_size((cover, secret))
            
            # Clip to valid range
            cover = torch.clamp(cover, -1, 1)
            secret = torch.clamp(secret, -1, 1)
            
            # Forward pass with mixed precision
            if self.config.training.use_mixed_precision:
                with autocast(dtype=torch.float16):
                    stego, recovered_secret, _ = self.model(cover, secret)
                    total_loss, loss_dict = self.loss_fn(
                        stego, cover, recovered_secret, secret
                    )
            else:
                stego, recovered_secret, _ = self.model(cover, secret)
                total_loss, loss_dict = self.loss_fn(
                    stego, cover, recovered_secret, secret
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.training.use_mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                total_loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_max_norm
                )
            
            # Optimizer step
            if self.config.training.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Track loss
            loss_tracker.update(loss_dict, batch_size=batch_size)
            
            # Log progress
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                avg_losses = loss_tracker.get_average()
                self.logger.log(
                    f"Epoch [{epoch+1}/{self.config.training.num_epochs}] "
                    f"Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {avg_losses.get('cover_reconstruction', 0):.4f}"
                )
        
        # Get epoch average
        epoch_metrics = loss_tracker.get_average()
        epoch_metrics["total_loss"] = (
            self.config.training.loss_weights["cover_reconstruction"] * epoch_metrics.get("cover_reconstruction", 0) +
            self.config.training.loss_weights["secret_reconstruction"] * epoch_metrics.get("secret_reconstruction", 0) +
            self.config.training.loss_weights["ssim"] * epoch_metrics.get("ssim", 0) +
            self.config.training.loss_weights["perceptual"] * epoch_metrics.get("perceptual", 0) +
            self.config.training.loss_weights["diffusion_consistency"] * epoch_metrics.get("diffusion_consistency", 0) +
            self.config.training.loss_weights["edge_preservation"] * epoch_metrics.get("edge_preservation", 0)
        )
        
        epoch_time = self.progress_tracker.end_epoch()
        self.logger.log(f"✓ Epoch {epoch+1} completed in {epoch_time}")
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        loss_tracker = LossTracker()
        metrics_tracker = MetricsTracker(self.config.validation)
        
        for batch in val_loader:
            cover, secret = BatchProcessor.move_batch_to_device(batch, self.device)
            cover = torch.clamp(cover, -1, 1)
            secret = torch.clamp(secret, -1, 1)
            batch_size = BatchProcessor.get_batch_size((cover, secret))
            
            # Forward pass
            stego, recovered_secret, _ = self.model(cover, secret)
            
            # Compute loss
            total_loss, loss_dict = self.loss_fn(
                stego, cover, recovered_secret, secret
            )
            loss_tracker.update(loss_dict, batch_size=batch_size)
            
            # Compute metrics
            metrics_tracker.update(stego, cover, recovered_secret, secret)
        
        # Combine results
        val_metrics = loss_tracker.get_average()
        val_metrics["total_loss"] = (
            self.config.training.loss_weights["cover_reconstruction"] * val_metrics.get("cover_reconstruction", 0) +
            self.config.training.loss_weights["secret_reconstruction"] * val_metrics.get("secret_reconstruction", 0) +
            self.config.training.loss_weights["ssim"] * val_metrics.get("ssim", 0) +
            self.config.training.loss_weights["perceptual"] * val_metrics.get("perceptual", 0) +
            self.config.training.loss_weights["diffusion_consistency"] * val_metrics.get("diffusion_consistency", 0) +
            self.config.training.loss_weights["edge_preservation"] * val_metrics.get("edge_preservation", 0)
        )
        
        metric_avgs = metrics_tracker.get_averages()
        val_metrics.update(metric_avgs)
        
        return val_metrics
    
    def train(self, train_loader, val_loader):
        """
        Complete training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        self.logger.log("\n" + "="*70)
        self.logger.log("🚀 STARTING TRAINING")
        self.logger.log("="*70 + "\n")
        
        # Initialize mixed precision scaler if needed
        if self.config.training.use_mixed_precision:
            self.scaler = GradScaler()
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase (every checkpoint interval)
            if (epoch + 1) % self.config.training.validation_interval == 0:
                self.logger.log(f"\n✔️  Validating at epoch {epoch+1}...")
                val_metrics = self.validate(val_loader)
                
                # Log metrics
                all_metrics = {"epoch": epoch + 1}
                all_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
                all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                self.metrics_logger.log_metrics(epoch + 1, all_metrics)
                
                # Print validation results
                self.logger.log(
                    f"\n📊 VALIDATION RESULTS - Epoch {epoch+1}\n"
                    f"  Train Loss: {train_metrics.get('total_loss', 0):.6f}\n"
                    f"  Val Loss:   {val_metrics.get('total_loss', 0):.6f}\n"
                    f"  Stego PSNR: {val_metrics.get('stego_psnr', 0):.4f} dB\n"
                    f"  Stego SSIM: {val_metrics.get('stego_ssim', 0):.4f}\n"
                    f"  Secret PSNR: {val_metrics.get('secret_psnr', 0):.4f} dB\n"
                    f"  BPP: {val_metrics.get('bpp', 0):.4f}\n"
                )
                
                # Save checkpoint
                is_best = val_metrics.get('total_loss', float('inf')) < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    val_metrics.get('total_loss', 0),
                    metrics={**train_metrics, **val_metrics},
                    config=self.config,
                    is_best=is_best
                )
                
                # Early stopping
                if (self.config.convergence.patience is not None and 
                    patience_counter >= self.config.convergence.patience):
                    self.logger.log(
                        f"\n⚠️  Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience_counter} epochs)"
                    )
                    break
            
            # Update scheduler
            self.scheduler.step()
            
            # Estimate time remaining
            if epoch > 0:
                time_remaining = self.progress_tracker.estimate_time_remaining(epoch)
                progress_bar = self.progress_tracker.get_progress_bar(epoch)
                self.logger.log(f"{progress_bar} | ETA: {time_remaining}")
        
        self.logger.log("\n" + "="*70)
        self.logger.log("✅ TRAINING COMPLETED")
        self.logger.log("="*70 + "\n")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train Diffusion-Based Image Steganography Model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG.training.num_epochs,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG.dataset.batch_size,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_CONFIG.training.learning_rate,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_imagenet",
        choices=["tiny_imagenet", "cifar10", "custom"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default=None,
        help="Path to custom dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("\n" + "="*70)
    print("🔧 SETTING UP TRAINING ENVIRONMENT")
    print("="*70 + "\n")
    
    # Set seed
    ReproducibilityManager.set_seed(
        DEFAULT_CONFIG.seed,
        DEFAULT_CONFIG.deterministic
    )
    
    # Get device
    device = DeviceManager.get_device(DEFAULT_CONFIG.use_gpu)
    
    # Create configuration
    config = DEFAULT_CONFIG
    config.training.num_epochs = args.epochs
    config.dataset.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.dataset.dataset_name = args.dataset
    config.training.use_mixed_precision = args.use_mixed_precision
    config.training.gradient_clip_max_norm = args.gradient_clip
    
    # Create logger
    log_file = config.logs_dir / "training.log"
    logger = Logger(log_file)
    
    # Print config
    logger.log(config.summary())
    logger.log(f"\n📁 Project root: {config.project_root}")
    logger.log(f"📁 Checkpoints: {config.checkpoints_dir}")
    logger.log(f"📁 Logs: {config.logs_dir}")
    logger.log(f"📁 Visualizations: {config.visualizations_dir}")
    
    # Create dataloaders
    logger.log("\n" + "="*70)
    logger.log("📂 LOADING DATASET")
    logger.log("="*70)
    
    train_loader, val_loader, dataset_info = create_dataloaders(
        config.dataset,
        dataset_name=args.dataset if args.dataset != "custom" else None,
        custom_dataset_path=args.custom_dataset
    )
    
    logger.log(f"\n✓ Dataset loaded successfully")
    for key, value in dataset_info.items():
        logger.log(f"  {key}: {value}")
    
    # Create trainer
    trainer = DiffusionSteganographyTrainer(config, device, logger)
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        logger.log(f"\n📂 Loading checkpoint: {args.checkpoint}")
        trainer.checkpoint_manager.load_checkpoint(
            args.checkpoint,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            device
        )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save final config
    ConfigManager.save_config(config, config.logs_dir / "final_config.json")
    logger.log(f"✓ Final config saved to {config.logs_dir / 'final_config.json'}")


if __name__ == "__main__":
    main()
