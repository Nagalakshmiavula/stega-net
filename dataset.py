"""
Dataset module for Diffusion-Based Image Steganography Model.

This module provides:
- Multiple image dataset loading (Tiny ImageNet, CIFAR-10, custom)
- Automatic image pairing strategy
- Data augmentation and normalization
- Train/validation splitting
- DataLoader creation
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import DatasetConfig, DEFAULT_CONFIG


# ============================================================================
# IMAGE PAIR DATASET
# ============================================================================

class ImagePairDataset(Dataset):
    """
    Dataset for image pairs (cover image, secret image).
    
    Images are automatically paired using a specified strategy.
    Each sample returns: (cover_image, secret_image)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        config: DatasetConfig,
        transform=None,
        pairing_strategy: str = "random"
    ):
        """
        Initialize ImagePairDataset.
        
        Args:
            image_paths: List of image file paths
            config: DatasetConfig instance
            transform: Albumentations transform pipeline
            pairing_strategy: How to pair images ("random", "sequential", "semantic")
        """
        self.image_paths = image_paths
        self.config = config
        self.transform = transform
        self.pairing_strategy = pairing_strategy
        self.num_images = len(image_paths)
        
        if self.num_images < 2:
            raise ValueError("Need at least 2 images for pairing")
        
        # Pre-generate pairs
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate image pairs based on strategy."""
        pairs = []
        
        if self.pairing_strategy == "random":
            # Random pairing: each image paired with random other image
            for i in range(self.num_images):
                # Pick a different random image as pair
                j = random.randint(0, self.num_images - 1)
                while j == i:  # Ensure different images
                    j = random.randint(0, self.num_images - 1)
                pairs.append((i, j))
        
        elif self.pairing_strategy == "sequential":
            # Sequential pairing: (0,1), (2,3), (4,5), ...
            for i in range(0, self.num_images - 1, 2):
                pairs.append((i, i + 1))
        
        elif self.pairing_strategy == "semantic":
            # Semantic pairing: similar images together (by filename sorting)
            sorted_indices = sorted(range(self.num_images), 
                                  key=lambda i: self.image_paths[i])
            for i in range(0, len(sorted_indices) - 1, 2):
                pairs.append((sorted_indices[i], sorted_indices[i + 1]))
        
        else:
            raise ValueError(f"Unknown pairing strategy: {self.pairing_strategy}")
        
        return pairs
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image pair at index.
        
        Returns:
            (cover_image, secret_image) as torch tensors
        """
        cover_idx, secret_idx = self.pairs[idx]
        
        # Load images
        cover_image = self._load_image(self.image_paths[cover_idx])
        secret_image = self._load_image(self.image_paths[secret_idx])
        
        # Apply transforms
        if self.transform:
            cover_image = self.transform(image=cover_image)["image"]
            secret_image = self.transform(image=secret_image)["image"]
        
        return cover_image, secret_image
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image and convert to RGB numpy array."""
        img = Image.open(path).convert("RGB")
        
        # Resize to target size
        img = img.resize(
            (self.config.image_width, self.config.image_height),
            Image.Resampling.LANCZOS
        )
        
        return np.array(img, dtype=np.uint8)


# ============================================================================
# TINY IMAGENET DATASET LOADER
# ============================================================================

class TinyImageNetLoader:
    """Load Tiny ImageNet dataset."""
    
    def __init__(self, data_dir: str = "data/tiny-imagenet-200"):
        """Initialize Tiny ImageNet loader."""
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
    
    def is_available(self) -> bool:
        """Check if Tiny ImageNet is available locally."""
        return self.train_dir.exists() and self.val_dir.exists()
    
    def get_image_paths(self) -> List[str]:
        """
        Get all image paths from Tiny ImageNet.
        
        Returns:
            List of image file paths
        """
        image_paths = []
        
        if not self.is_available():
            print(f"⚠️  Tiny ImageNet not found at {self.data_dir}")
            print("Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            return image_paths
        
        # Get training images
        for class_dir in self.train_dir.iterdir():
            if class_dir.is_dir():
                images_dir = class_dir / "images"
                if images_dir.exists():
                    for img_file in images_dir.glob("*.JPEG"):
                        image_paths.append(str(img_file))
        
        print(f"✓ Loaded {len(image_paths)} images from Tiny ImageNet")
        return image_paths


# ============================================================================
# CUSTOM DATASET LOADER
# ============================================================================

class CustomDatasetLoader:
    """Load images from custom directory."""
    
    def __init__(self, data_dir: str):
        """Initialize custom dataset loader."""
        self.data_dir = Path(data_dir)
    
    def is_available(self) -> bool:
        """Check if directory exists."""
        return self.data_dir.exists()
    
    def get_image_paths(self) -> List[str]:
        """
        Get all image paths from custom directory.
        
        Supports: .jpg, .jpeg, .png, .bmp, .tiff
        
        Returns:
            List of image file paths
        """
        if not self.is_available():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPEG"}
        image_paths = []
        
        # Recursively search for images
        for img_file in self.data_dir.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                image_paths.append(str(img_file))
        
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.data_dir}\n"
                f"Supported formats: {', '.join(image_extensions)}"
            )
        
        print(f"✓ Loaded {len(image_paths)} images from {self.data_dir}")
        return image_paths


# ============================================================================
# CIFAR-10 DATASET LOADER
# ============================================================================

class CIFAR10Loader:
    """Load CIFAR-10 dataset from torchvision."""
    
    def __init__(self, data_dir: str = "data/cifar10"):
        """Initialize CIFAR-10 loader."""
        self.data_dir = data_dir
    
    def get_image_paths(self) -> List[str]:
        """
        Download and extract CIFAR-10, return image paths.
        
        Returns:
            List of image file paths
        """
        # Create directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Download CIFAR-10
        cifar10 = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None
        )
        
        # Save images to disk
        save_dir = Path(self.data_dir) / "images"
        save_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for idx, (img, label) in enumerate(cifar10):
            img_path = save_dir / f"cifar10_{idx:06d}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
        
        print(f"✓ Loaded {len(image_paths)} images from CIFAR-10")
        return image_paths


# ============================================================================
# DATA AUGMENTATION PIPELINE
# ============================================================================

def get_augmentation_pipeline(config: DatasetConfig) -> Optional[A.Compose]:
    """
    Create Albumentations augmentation pipeline.
    
    Args:
        config: DatasetConfig instance
    
    Returns:
        Albumentations Compose pipeline or None
    """
    if not config.enable_augmentation:
        return None
    
    transforms_list = [
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        
        # Color augmentations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussBlur(p=0.2),
        
        # Normalization
        A.Normalize(
            mean=config.norm_mean,
            std=config.norm_std,
            always_apply=True
        ),
        
        # Convert to tensor
        ToTensorV2(always_apply=True)
    ]
    
    return A.Compose(transforms_list, p=config.augmentation_probability)


def get_validation_pipeline(config: DatasetConfig) -> A.Compose:
    """
    Create validation augmentation pipeline (only normalization).
    
    Args:
        config: DatasetConfig instance
    
    Returns:
        Albumentations Compose pipeline
    """
    transforms_list = [
        A.Normalize(
            mean=config.norm_mean,
            std=config.norm_std,
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ]
    
    return A.Compose(transforms_list)


# ============================================================================
# DATASET CREATION & DATALOADER
# ============================================================================

def create_dataloaders(
    config: DatasetConfig,
    dataset_name: Optional[str] = None,
    custom_dataset_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation dataloaders.
    
    Args:
        config: DatasetConfig instance
        dataset_name: Override config dataset name
        custom_dataset_path: Override custom dataset path
    
    Returns:
        (train_loader, val_loader, dataset_info_dict)
    """
    
    # Determine dataset source
    if custom_dataset_path:
        loader = CustomDatasetLoader(custom_dataset_path)
    elif dataset_name == "tiny_imagenet":
        loader = TinyImageNetLoader()
    elif dataset_name == "cifar10":
        loader = CIFAR10Loader()
    else:
        loader = TinyImageNetLoader()  # Default
    
    # Get image paths
    if isinstance(loader, TinyImageNetLoader) and not loader.is_available():
        print("\n" + "="*70)
        print("📥 Tiny ImageNet not found. Downloading CIFAR-10 instead...")
        print("="*70)
        loader = CIFAR10Loader()
    
    image_paths = loader.get_image_paths()
    
    if not image_paths:
        raise RuntimeError(
            "No images found. Please ensure dataset is available or "
            "provide custom dataset path."
        )
    
    # Shuffle image paths
    random.shuffle(image_paths)
    
    # Split into train and validation
    split_idx = int(len(image_paths) * config.train_val_split)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   • Training images: {len(train_paths)}")
    print(f"   • Validation images: {len(val_paths)}")
    print(f"   • Pairing strategy: {config.pairing_strategy}")
    
    # Create augmentation pipelines
    train_transform = get_augmentation_pipeline(config)
    val_transform = get_validation_pipeline(config)
    
    # Create datasets
    train_dataset = ImagePairDataset(
        train_paths,
        config,
        transform=train_transform,
        pairing_strategy=config.pairing_strategy
    )
    
    val_dataset = ImagePairDataset(
        val_paths,
        config,
        transform=val_transform,
        pairing_strategy=config.pairing_strategy
    )
    
    print(f"   • Training pairs: {len(train_dataset)}")
    print(f"   • Validation pairs: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    dataset_info = {
        "total_images": len(image_paths),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "batch_size": config.batch_size,
        "image_size": (config.image_height, config.image_width, config.image_channels),
        "num_workers": config.num_workers,
        "augmentation_enabled": config.enable_augmentation,
        "pairing_strategy": config.pairing_strategy,
    }
    
    return train_loader, val_loader, dataset_info


# ============================================================================
# DATASET STATISTICS & VISUALIZATION
# ============================================================================

def compute_dataset_statistics(
    loader: DataLoader,
    num_batches: Optional[int] = None
) -> Dict:
    """
    Compute statistics (mean, std) of dataset.
    
    Args:
        loader: DataLoader instance
        num_batches: Number of batches to compute from (None = all)
    
    Returns:
        Dictionary with statistics
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for batch_idx, (cover, secret) in enumerate(loader):
        if num_batches and batch_idx >= num_batches:
            break
        
        # Process cover images
        batch_mean = cover.mean(dim=(0, 2, 3))
        batch_std = cover.std(dim=(0, 2, 3))
        
        mean += batch_mean * cover.shape[0]
        std += batch_std * cover.shape[0]
        total_samples += cover.shape[0]
    
    mean /= total_samples
    std /= total_samples
    
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_samples": total_samples
    }


def visualize_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    save_path: Optional[str] = None,
    denormalize: bool = True
) -> None:
    """
    Visualize a batch of image pairs.
    
    Args:
        batch: (cover_images, secret_images) tuple
        save_path: Path to save visualization
        denormalize: Whether to denormalize from ImageNet stats
    """
    import matplotlib.pyplot as plt
    
    cover_batch, secret_batch = batch
    batch_size = min(4, cover_batch.shape[0])
    
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Denormalize
        cover = cover_batch[i].permute(1, 2, 0).numpy()
        secret = secret_batch[i].permute(1, 2, 0).numpy()
        
        if denormalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            cover = (cover * std + mean).clip(0, 1)
            secret = (secret * std + mean).clip(0, 1)
        
        axes[i, 0].imshow(cover)
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(secret)
        axes[i, 1].set_title("Secret Image")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved batch visualization to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_dataset():
    """Test dataset loading and visualization."""
    print("\n" + "="*70)
    print("🧪 DATASET TESTING")
    print("="*70)
    
    # Create dataloaders
    config = DEFAULT_CONFIG.dataset
    
    try:
        train_loader, val_loader, dataset_info = create_dataloaders(config)
        
        print("\n✅ DataLoaders created successfully!")
        print("\n📈 Dataset Info:")
        for key, value in dataset_info.items():
            print(f"   • {key}: {value}")
        
        # Get first batch
        print("\n📦 Getting first batch...")
        cover_batch, secret_batch = next(iter(train_loader))
        print(f"   • Cover shape: {cover_batch.shape}")
        print(f"   • Secret shape: {secret_batch.shape}")
        print(f"   • Dtype: {cover_batch.dtype}")
        print(f"   • Min/Max values: [{cover_batch.min():.3f}, {cover_batch.max():.3f}]")
        
        # Compute statistics
        print("\n📊 Computing dataset statistics...")
        stats = compute_dataset_statistics(train_loader, num_batches=10)
        print(f"   • Mean: {stats['mean']}")
        print(f"   • Std: {stats['std']}")
        
        # Visualize batch
        print("\n🖼️  Visualizing batch...")
        visualize_batch((cover_batch, secret_batch))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
