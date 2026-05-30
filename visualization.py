"""
Visualization Module for Training Analysis and Results.

This module implements:
- Loss curves (training vs validation)
- PSNR/SSIM convergence plots
- BPP analysis charts
- Convergence reports
- Sample image grids (cover, stego, secret, recovered)
- Heatmaps and statistical plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

from config import FullConfig


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

class PlotStyle:
    """Consistent plotting style."""
    
    # Colors
    COLOR_TRAIN = "#2E86AB"      # Blue
    COLOR_VAL = "#A23B72"        # Purple
    COLOR_LOSS = "#F18F01"       # Orange
    COLOR_SSIM = "#C73E1D"       # Red
    COLOR_PSNR = "#6A994E"       # Green
    COLOR_BPP = "#BC4749"        # Dark red
    
    # Fonts
    FONT_LARGE = 16
    FONT_MEDIUM = 12
    FONT_SMALL = 10
    
    @staticmethod
    def setup():
        """Set up matplotlib style."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.rcParams["font.size"] = PlotStyle.FONT_MEDIUM
        plt.rcParams["lines.linewidth"] = 2.5
        plt.rcParams["lines.markersize"] = 8


# ============================================================================
# LOSS AND METRICS VISUALIZATION
# ============================================================================

class LossVisualizer:
    """Visualize training/validation loss curves."""
    
    @staticmethod
    def plot_loss_curves(
        metrics_df: pd.DataFrame,
        output_path: Path,
        title: str = "Training and Validation Loss"
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            metrics_df: Pandas DataFrame with metrics
            output_path: Path to save figure
            title: Plot title
        """
        PlotStyle.setup()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract columns
        if "train_total_loss" in metrics_df.columns:
            ax.plot(
                metrics_df["epoch"],
                metrics_df["train_total_loss"],
                label="Training Loss",
                color=PlotStyle.COLOR_TRAIN,
                marker="o",
                markersize=4,
                alpha=0.8
            )
        
        if "val_total_loss" in metrics_df.columns:
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_total_loss"],
                label="Validation Loss",
                color=PlotStyle.COLOR_VAL,
                marker="s",
                markersize=4,
                alpha=0.8
            )
        
        ax.set_xlabel("Epoch", fontsize=PlotStyle.FONT_LARGE)
        ax.set_ylabel("Loss", fontsize=PlotStyle.FONT_LARGE)
        ax.set_title(title, fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
        ax.legend(fontsize=PlotStyle.FONT_MEDIUM, loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Loss curves saved: {output_path}")
    
    @staticmethod
    def plot_loss_components(
        metrics_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Plot individual loss components over time.
        
        Args:
            metrics_df: Pandas DataFrame with metrics
            output_path: Path to save figure
        """
        PlotStyle.setup()
        
        # Get loss component columns
        loss_components = [
            col for col in metrics_df.columns
            if "val_" in col and col.endswith("_loss")
        ]
        
        if not loss_components:
            print("⚠️  No loss components found in metrics")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(loss_components)))
        
        for idx, component in enumerate(loss_components):
            ax = axes[idx]
            
            ax.plot(
                metrics_df["epoch"],
                metrics_df[component],
                color=colors[idx],
                marker="o",
                markersize=5,
                linewidth=2.5
            )
            
            component_name = component.replace("val_", "").replace("_", " ").title()
            ax.set_title(component_name, fontsize=PlotStyle.FONT_MEDIUM, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=PlotStyle.FONT_SMALL)
            ax.set_ylabel("Loss", fontsize=PlotStyle.FONT_SMALL)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(loss_components), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Loss components saved: {output_path}")


class MetricsVisualizer:
    """Visualize quality metrics (PSNR, SSIM, BPP)."""
    
    @staticmethod
    def plot_psnr_ssim(
        metrics_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Plot PSNR and SSIM for stego and secret images.
        
        Args:
            metrics_df: Pandas DataFrame with metrics
            output_path: Path to save figure
        """
        PlotStyle.setup()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Stego PSNR
        if "val_stego_psnr" in metrics_df.columns:
            ax = axes[0, 0]
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_stego_psnr"],
                color=PlotStyle.COLOR_PSNR,
                marker="o",
                markersize=6,
                linewidth=2.5
            )
            ax.fill_between(
                metrics_df["epoch"],
                metrics_df["val_stego_psnr"] - 2,
                metrics_df["val_stego_psnr"] + 2,
                alpha=0.2,
                color=PlotStyle.COLOR_PSNR
            )
            ax.axhline(y=35, color="red", linestyle="--", label="Good Quality (35 dB)")
            ax.set_title("Stego Image PSNR", fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
            ax.set_ylabel("PSNR (dB)", fontsize=PlotStyle.FONT_MEDIUM)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Stego SSIM
        if "val_stego_ssim" in metrics_df.columns:
            ax = axes[0, 1]
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_stego_ssim"],
                color=PlotStyle.COLOR_SSIM,
                marker="s",
                markersize=6,
                linewidth=2.5
            )
            ax.fill_between(
                metrics_df["epoch"],
                metrics_df["val_stego_ssim"] - 0.01,
                metrics_df["val_stego_ssim"] + 0.01,
                alpha=0.2,
                color=PlotStyle.COLOR_SSIM
            )
            ax.axhline(y=0.95, color="green", linestyle="--", label="Good Quality (0.95)")
            ax.set_title("Stego Image SSIM", fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
            ax.set_ylabel("SSIM", fontsize=PlotStyle.FONT_MEDIUM)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Secret PSNR
        if "val_secret_psnr" in metrics_df.columns:
            ax = axes[1, 0]
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_secret_psnr"],
                color=PlotStyle.COLOR_PSNR,
                marker="o",
                markersize=6,
                linewidth=2.5
            )
            ax.axhline(y=30, color="orange", linestyle="--", label="Good Recovery (30 dB)")
            ax.set_title("Secret Image PSNR", fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=PlotStyle.FONT_MEDIUM)
            ax.set_ylabel("PSNR (dB)", fontsize=PlotStyle.FONT_MEDIUM)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Secret SSIM
        if "val_secret_ssim" in metrics_df.columns:
            ax = axes[1, 1]
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_secret_ssim"],
                color=PlotStyle.COLOR_SSIM,
                marker="s",
                markersize=6,
                linewidth=2.5
            )
            ax.axhline(y=0.90, color="green", linestyle="--", label="Good Recovery (0.90)")
            ax.set_title("Secret Image SSIM", fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=PlotStyle.FONT_MEDIUM)
            ax.set_ylabel("SSIM", fontsize=PlotStyle.FONT_MEDIUM)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ PSNR/SSIM curves saved: {output_path}")
    
    @staticmethod
    def plot_bpp_analysis(
        metrics_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Plot BPP and capacity analysis.
        
        Args:
            metrics_df: Pandas DataFrame with metrics
            output_path: Path to save figure
        """
        PlotStyle.setup()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if "val_bpp" in metrics_df.columns:
            ax.plot(
                metrics_df["epoch"],
                metrics_df["val_bpp"],
                color=PlotStyle.COLOR_BPP,
                marker="D",
                markersize=7,
                linewidth=2.5,
                label="BPP"
            )
            
            # Fill area
            ax.fill_between(
                metrics_df["epoch"],
                metrics_df["val_bpp"],
                alpha=0.2,
                color=PlotStyle.COLOR_BPP
            )
            
            # Capacity targets
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="High Capacity (1.0)")
            ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Medium Capacity (0.5)")
            ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.7, label="Low Capacity (0.1)")
            
            ax.set_xlabel("Epoch", fontsize=PlotStyle.FONT_LARGE)
            ax.set_ylabel("BPP (Bits Per Pixel)", fontsize=PlotStyle.FONT_LARGE)
            ax.set_title("Embedding Capacity (BPP) Analysis", fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
            ax.legend(fontsize=PlotStyle.FONT_MEDIUM, loc="best")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ BPP analysis saved: {output_path}")


# ============================================================================
# IMAGE VISUALIZATION
# ============================================================================

class ImageVisualizer:
    """Visualize sample images."""
    
    @staticmethod
    def plot_sample_grid(
        covers: torch.Tensor,
        stegos: torch.Tensor,
        secrets: torch.Tensor,
        recovered_secrets: torch.Tensor,
        output_path: Path,
        num_samples: int = 4,
        epoch: int = 0
    ):
        """
        Plot grid of sample images (cover, stego, secret, recovered).
        
        Args:
            covers: Cover images (B, 3, H, W)
            stegos: Stego images (B, 3, H, W)
            secrets: Secret images (B, 3, H, W)
            recovered_secrets: Recovered secret images (B, 3, H, W)
            output_path: Path to save figure
            num_samples: Number of samples to show
            epoch: Epoch number for title
        """
        PlotStyle.setup()
        
        num_samples = min(num_samples, covers.shape[0])
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_samples):
            # Convert to numpy and denormalize
            cover_img = ImageVisualizer._to_pil(covers[idx])
            stego_img = ImageVisualizer._to_pil(stegos[idx])
            secret_img = ImageVisualizer._to_pil(secrets[idx])
            recovered_img = ImageVisualizer._to_pil(recovered_secrets[idx])
            
            # Cover
            axes[idx, 0].imshow(cover_img)
            axes[idx, 0].set_title("Cover Image", fontsize=PlotStyle.FONT_MEDIUM, fontweight="bold")
            axes[idx, 0].axis("off")
            
            # Stego
            axes[idx, 1].imshow(stego_img)
            axes[idx, 1].set_title("Stego Image", fontsize=PlotStyle.FONT_MEDIUM, fontweight="bold")
            axes[idx, 1].axis("off")
            
            # Secret
            axes[idx, 2].imshow(secret_img)
            axes[idx, 2].set_title("Secret Image", fontsize=PlotStyle.FONT_MEDIUM, fontweight="bold")
            axes[idx, 2].axis("off")
            
            # Recovered
            axes[idx, 3].imshow(recovered_img)
            axes[idx, 3].set_title("Recovered Secret", fontsize=PlotStyle.FONT_MEDIUM, fontweight="bold")
            axes[idx, 3].axis("off")
        
        fig.suptitle(
            f"Sample Images - Epoch {epoch}",
            fontsize=PlotStyle.FONT_LARGE,
            fontweight="bold",
            y=0.995
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Sample grid saved: {output_path}")
    
    @staticmethod
    def _to_pil(tensor: torch.Tensor):
        """Convert tensor to PIL image."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach().numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = np.clip(tensor, 0, 1)
        
        # Convert from CHW to HWC
        if tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))
        
        # Convert to uint8
        tensor = (tensor * 255).astype(np.uint8)
        
        from PIL import Image
        return Image.fromarray(tensor)
    
    @staticmethod
    def plot_difference_map(
        img1: torch.Tensor,
        img2: torch.Tensor,
        output_path: Path,
        title: str = "Difference Map"
    ):
        """
        Plot difference map between two images.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            output_path: Path to save figure
            title: Plot title
        """
        PlotStyle.setup()
        
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().detach().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().detach().numpy()
        
        # Compute absolute difference
        diff = np.abs(img1 - img2)
        
        # Take mean across channels
        if diff.ndim == 3:
            diff = np.mean(diff, axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(diff, cmap="hot")
        ax.set_title(title, fontsize=PlotStyle.FONT_LARGE, fontweight="bold")
        ax.axis("off")
        
        cbar = plt.colorbar(im, ax=ax, label="Absolute Difference")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Difference map saved: {output_path}")


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

class ConvergenceAnalyzer:
    """Analyze and report convergence."""
    
    @staticmethod
    def plot_convergence_report(
        metrics_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Create comprehensive convergence report.
        
        Args:
            metrics_df: Pandas DataFrame with metrics
            output_path: Path to save figure
        """
        PlotStyle.setup()
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss convergence
        ax1 = fig.add_subplot(gs[0, :2])
        if "train_total_loss" in metrics_df.columns and "val_total_loss" in metrics_df.columns:
            ax1.plot(metrics_df["epoch"], metrics_df["train_total_loss"], label="Train", linewidth=2)
            ax1.plot(metrics_df["epoch"], metrics_df["val_total_loss"], label="Val", linewidth=2)
            ax1.set_title("Loss Convergence", fontweight="bold", fontsize=PlotStyle.FONT_LARGE)
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Key metrics
        ax2 = fig.add_subplot(gs[0, 2])
        metrics_text = "📊 FINAL METRICS\n" + "="*25 + "\n"
        if "val_stego_psnr" in metrics_df.columns:
            final_psnr = metrics_df["val_stego_psnr"].iloc[-1]
            metrics_text += f"Stego PSNR: {final_psnr:.2f} dB\n"
        if "val_stego_ssim" in metrics_df.columns:
            final_ssim = metrics_df["val_stego_ssim"].iloc[-1]
            metrics_text += f"Stego SSIM: {final_ssim:.4f}\n"
        if "val_bpp" in metrics_df.columns:
            final_bpp = metrics_df["val_bpp"].iloc[-1]
            metrics_text += f"BPP: {final_bpp:.4f}\n"
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=PlotStyle.FONT_SMALL, family="monospace",
                verticalalignment="center")
        ax2.axis("off")
        
        # 3. PSNR convergence
        ax3 = fig.add_subplot(gs[1, 0])
        if "val_stego_psnr" in metrics_df.columns:
            ax3.plot(metrics_df["epoch"], metrics_df["val_stego_psnr"], color=PlotStyle.COLOR_PSNR)
            ax3.fill_between(metrics_df["epoch"], metrics_df["val_stego_psnr"], alpha=0.3)
            ax3.set_title("Stego PSNR", fontweight="bold")
            ax3.set_ylabel("PSNR (dB)")
            ax3.grid(True, alpha=0.3)
        
        # 4. SSIM convergence
        ax4 = fig.add_subplot(gs[1, 1])
        if "val_stego_ssim" in metrics_df.columns:
            ax4.plot(metrics_df["epoch"], metrics_df["val_stego_ssim"], color=PlotStyle.COLOR_SSIM)
            ax4.fill_between(metrics_df["epoch"], metrics_df["val_stego_ssim"], alpha=0.3)
            ax4.set_title("Stego SSIM", fontweight="bold")
            ax4.set_ylabel("SSIM")
            ax4.grid(True, alpha=0.3)
        
        # 5. BPP convergence
        ax5 = fig.add_subplot(gs[1, 2])
        if "val_bpp" in metrics_df.columns:
            ax5.plot(metrics_df["epoch"], metrics_df["val_bpp"], color=PlotStyle.COLOR_BPP)
            ax5.fill_between(metrics_df["epoch"], metrics_df["val_bpp"], alpha=0.3)
            ax5.set_title("Embedding Capacity (BPP)", fontweight="bold")
            ax5.set_ylabel("BPP")
            ax5.grid(True, alpha=0.3)
        
        # 6. Secret PSNR convergence
        ax6 = fig.add_subplot(gs[2, 0])
        if "val_secret_psnr" in metrics_df.columns:
            ax6.plot(metrics_df["epoch"], metrics_df["val_secret_psnr"], color=PlotStyle.COLOR_PSNR, linestyle="--")
            ax6.fill_between(metrics_df["epoch"], metrics_df["val_secret_psnr"], alpha=0.3)
            ax6.set_title("Secret PSNR", fontweight="bold")
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("PSNR (dB)")
            ax6.grid(True, alpha=0.3)
        
        # 7. Secret SSIM convergence
        ax7 = fig.add_subplot(gs[2, 1])
        if "val_secret_ssim" in metrics_df.columns:
            ax7.plot(metrics_df["epoch"], metrics_df["val_secret_ssim"], color=PlotStyle.COLOR_SSIM, linestyle="--")
            ax7.fill_between(metrics_df["epoch"], metrics_df["val_secret_ssim"], alpha=0.3)
            ax7.set_title("Secret SSIM", fontweight="bold")
            ax7.set_xlabel("Epoch")
            ax7.set_ylabel("SSIM")
            ax7.grid(True, alpha=0.3)
        
        # 8. Convergence summary
        ax8 = fig.add_subplot(gs[2, 2])
        summary_text = "📈 CONVERGENCE\n" + "="*25 + "\n"
        if "val_total_loss" in metrics_df.columns:
            first_loss = metrics_df["val_total_loss"].iloc[0]
            last_loss = metrics_df["val_total_loss"].iloc[-1]
            improvement = ((first_loss - last_loss) / first_loss) * 100
            summary_text += f"Loss Improvement:\n{improvement:.1f}%\n\n"
        
        summary_text += f"Epochs: {len(metrics_df)}\n"
        summary_text += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        
        ax8.text(0.1, 0.5, summary_text, fontsize=PlotStyle.FONT_SMALL, family="monospace",
                verticalalignment="center")
        ax8.axis("off")
        
        fig.suptitle("Training Convergence Report", fontsize=PlotStyle.FONT_LARGE, fontweight="bold", y=0.995)
        
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Convergence report saved: {output_path}")


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def generate_all_visualizations(
    config: FullConfig,
    metrics_csv: Path,
    sample_images: Optional[Tuple] = None,
    epoch: int = 0
):
    """
    Generate all visualizations.
    
    Args:
        config: Full configuration
        metrics_csv: Path to metrics CSV file
        sample_images: Tuple of (covers, stegos, secrets, recovered_secrets)
        epoch: Epoch number
    """
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Create output directory
    viz_dir = config.visualizations_dir
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics if available
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        
        # Plot loss curves
        LossVisualizer.plot_loss_curves(
            metrics_df,
            viz_dir / "loss_curves.png"
        )
        
        # Plot loss components
        LossVisualizer.plot_loss_components(
            metrics_df,
            viz_dir / "loss_components.png"
        )
        
        # Plot PSNR and SSIM
        MetricsVisualizer.plot_psnr_ssim(
            metrics_df,
            viz_dir / "psnr_ssim_curves.png"
        )
        
        # Plot BPP analysis
        MetricsVisualizer.plot_bpp_analysis(
            metrics_df,
            viz_dir / "bpp_analysis.png"
        )
        
        # Plot convergence report
        ConvergenceAnalyzer.plot_convergence_report(
            metrics_df,
            viz_dir / "convergence_report.png"
        )
    
    # Plot sample images if provided
    if sample_images is not None:
        covers, stegos, secrets, recovered_secrets = sample_images
        
        ImageVisualizer.plot_sample_grid(
            covers,
            stegos,
            secrets,
            recovered_secrets,
            viz_dir / f"sample_grids_epoch_{epoch:04d}.png",
            num_samples=4,
            epoch=epoch
        )
    
    print("\n✓ All visualizations completed!")


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
