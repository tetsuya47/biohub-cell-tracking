"""Module for PyTorch Dataset wrapper for Biohub Cell Tracking datasets."""

from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import Dataset

from src.datasets.biohub_dataset import BiohubDataset


class BiohubTorchDataset(Dataset):
    """PyTorch Dataset wrapper for Biohub Cell Tracking data.

    Utilizes BiohubDataset internally and converts all returned NumPy arrays
    to PyTorch Tensors using torch.from_numpy().
    """

    def __init__(self, zarr_path: Union[str, Path], geff_path: Union[str, Path]) -> None:
        """Initialize the PyTorch Dataset.

        Args:
            zarr_path: Path to the .zarr image dataset directory.
            geff_path: Path to the .geff ground truth dataset directory.
        """
        self.dataset = BiohubDataset(zarr_path, geff_path)

    def __len__(self) -> int:
        """Get the total number of timepoints in the dataset.

        Returns:
            int: Number of timepoints (T).
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get PyTorch Tensors for a specific timepoint index.

        Args:
            index: Timepoint index. Supports negative indexing.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing exactly:
                - "image": torch.Tensor of shape (Z, Y, X) for the given timepoint.
                - "node_ids": torch.Tensor of shape (N_t,) containing node IDs.
                - "positions": torch.Tensor of shape (N_t, 4) containing [t, z, y, x] positions.
                - "edges": torch.Tensor of shape (E_t, 2) containing directed edges.

        Raises:
            IndexError: If index is out of bounds.
        """
        sample = self.dataset[index]
        return {
            "image": torch.from_numpy(sample["image"]),
            "node_ids": torch.from_numpy(sample["node_ids"]),
            "positions": torch.from_numpy(sample["positions"]),
            "edges": torch.from_numpy(sample["edges"]),
        }
