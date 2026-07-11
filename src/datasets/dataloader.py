"""Module for PyTorch DataLoader creation for Biohub Cell Tracking datasets."""

from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.torch_dataset import BiohubTorchDataset
from src.preprocessing.transforms import TransformPipeline
from src.preprocessing.target_generator import TargetGenerator


class TransformedDataset(Dataset):
    """Dataset wrapper that applies the pre-processing TransformPipeline and TargetGenerator."""

    def __init__(self, ds: BiohubTorchDataset, tf: TransformPipeline, tg: TargetGenerator) -> None:
        """Initialize the TransformedDataset.

        Args:
            ds: Base BiohubTorchDataset instance.
            tf: Pre-processing TransformPipeline instance.
            tg: TargetGenerator instance for ground truth heatmaps.
        """
        self.ds = ds
        self.tf = tf
        self.tg = tg

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get transformed sample.

        Args:
            idx: Index of sample.

        Returns:
            Dict[str, torch.Tensor]: Transformed sample containing scaled image,
                tensors for node IDs, positions, edges, and generated target heatmap.
        """
        sample = self.ds[idx]
        transformed_sample = self.tf(sample)
        
        # Extract positions to generate dense supervision heatmap target
        positions = transformed_sample["positions"]
        target = self.tg.generate(positions)
        
        transformed_sample["target"] = target
        return transformed_sample


class BiohubDataLoader:
    """DataLoader builder for Biohub Cell Tracking data.

    Instantiates BiohubTorchDataset internally, applies the pre-processing
    TransformPipeline and TargetGenerator, and returns a configured PyTorch DataLoader.
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        geff_path: Union[str, Path],
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        """Initialize the BiohubDataLoader configuration parameters.

        Args:
            zarr_path: Path to the .zarr image dataset directory.
            geff_path: Path to the .geff ground truth dataset directory.
            batch_size: Batch size of the DataLoader.
            shuffle: Whether to shuffle the data.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster host-to-device transfers.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.zarr_path = zarr_path
        self.geff_path = geff_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def get_train_loader(self) -> DataLoader:
        """Create and configure the training PyTorch DataLoader.

        Applies the data TransformPipeline and TargetGenerator.

        Returns:
            DataLoader: Configured PyTorch DataLoader.
        """
        # Load raw dataset internally
        raw_dataset = BiohubTorchDataset(self.zarr_path, self.geff_path)
        
        # Get shape dynamically from dataset reader
        shape = raw_dataset.dataset.zarr_reader.get_shape()
        volume_shape = shape[1:]
        
        target_generator = TargetGenerator(volume_shape=volume_shape)
        pipeline = TransformPipeline()

        transformed_dataset = TransformedDataset(raw_dataset, pipeline, target_generator)

        return DataLoader(
            transformed_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
