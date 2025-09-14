import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, List
import glob

class MultiChannelDataset(Dataset):
    """
    Dataset for loading 15-channel images with 460x460 resolution.
    Supports multiple file formats: .npy, .tiff, .h5, etc.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        file_extension: str = 'npy',
        channels: int = 15,
        image_size: tuple = (460, 460)
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.image_size = image_size

        # Find all image files
        pattern = os.path.join(root_dir, f"*.{file_extension}")
        self.image_files = glob.glob(pattern)

        if len(self.image_files) == 0:
            raise ValueError(f"No {file_extension} files found in {root_dir}")

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_files[idx]

        # Load the 15-channel data
        if img_path.endswith('.npy'):
            image = np.load(img_path)
        elif img_path.endswith('.tiff') or img_path.endswith('.tif'):
            # Use tifffile for multi-channel TIFF support
            try:
                import tifffile
                image = tifffile.imread(img_path)
            except ImportError:
                # Fallback to PIL for single-channel TIFFs
                from PIL import Image
                image = np.array(Image.open(img_path))
        else:
            raise ValueError(f"Unsupported file format: {img_path}")

        # Ensure correct shape: (channels, height, width)
        if len(image.shape) == 3:
            if image.shape[2] == self.channels:  # (H, W, C)
                image = np.transpose(image, (2, 0, 1))  # -> (C, H, W)
            elif image.shape[0] == self.channels:  # Already (C, H, W)
                pass  # Shape is already correct
            else:
                raise ValueError(f"Expected {self.channels} channels, got {image.shape}")
        else:
            raise ValueError(f"Expected 3D array, got shape {image.shape}")

        # Ensure correct image size
        if image.shape[1:] != self.image_size:
            # Simple resize using numpy (for more advanced resizing, use cv2 or PIL)
            from scipy.ndimage import zoom
            h_factor = self.image_size[0] / image.shape[1]
            w_factor = self.image_size[1] / image.shape[2]
            image = zoom(image, (1, h_factor, w_factor), order=1)

        # Normalize to [0, 1] if values are in [0, 255] range
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image.copy()).float()

        # For SSL, we only need the image (label is dummy)
        return image_tensor, 0


class MultiChannelH5Dataset(Dataset):
    """
    Alternative dataset implementation for HDF5 files.
    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        h5_file_path: str,
        dataset_key: str = 'images',
        transform: Optional[Callable] = None,
        channels: int = 15
    ):
        import h5py
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key
        self.transform = transform
        self.channels = channels

        # Open file to get dataset info
        with h5py.File(h5_file_path, 'r') as f:
            self.dataset_length = len(f[dataset_key])
            self.dataset_shape = f[dataset_key].shape[1:]  # (C, H, W)

        print(f"H5 Dataset: {self.dataset_length} images, shape: {self.dataset_shape}")

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple:
        import h5py

        with h5py.File(self.h5_file_path, 'r') as f:
            image = f[self.dataset_key][idx]

        # Convert to tensor
        image_tensor = torch.from_numpy(image.copy()).float()

        # Normalize if needed
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        return image_tensor, 0