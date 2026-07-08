"""Module for unified BiohubDataset combining Zarr image volumes and GEFF ground truth tracking graphs."""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from src.datasets.geff_reader import GEFFReader
from src.datasets.zarr_dataset import ZarrDatasetReader


class BiohubDataset:
    """Unified dataset combining raw Zarr image volumes and GEFF ground truth tracking graph data."""

    def __init__(self, zarr_path: Union[str, Path], geff_path: Union[str, Path]) -> None:
        """Initialize the BiohubDataset.

        Args:
            zarr_path: Path to the .zarr image dataset directory.
            geff_path: Path to the .geff ground truth dataset directory.
        """
        self.zarr_reader = ZarrDatasetReader(zarr_path)
        self.geff_reader = GEFFReader(geff_path)

    def __len__(self) -> int:
        """Get the total number of timepoints in the dataset.

        Returns:
            int: Number of timepoints (T).
        """
        return self.zarr_reader.get_shape()[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get the image volume and ground truth tracking graph data for a specific timepoint.

        Args:
            index: Timepoint index. Supports negative indexing.

        Returns:
            Dict[str, Any]: Dictionary containing exactly the following keys:
                - "image": 3D numpy array of shape (Z, Y, X) for the given timepoint.
                - "node_ids": 1D numpy array of node IDs (uint64) at this timepoint.
                - "positions": 2D numpy array of shape (N_t, 4) containing [t, z, y, x] positions.
                - "edges": 2D numpy array of shape (E_t, 2) containing directed edges originating at this timepoint.

        Raises:
            IndexError: If index is out of bounds.
        """
        # Normalize and validate index
        T = len(self)
        if index < 0:
            index += T

        if index < 0 or index >= T:
            raise IndexError(f"Timepoint index {index} out of bounds for dataset of length {T}")

        # Load 3D image volume
        image = self.zarr_reader.get_frame(index)

        # Load nodes and positions
        all_ids = self.geff_reader.get_node_ids()
        all_pos = self.geff_reader.get_node_positions()

        # Filter nodes at this timepoint
        mask = all_pos[:, 0] == index
        node_ids = all_ids[mask]
        positions = all_pos[mask]

        # Filter edges originating at this timepoint
        all_edges = self.geff_reader.get_edges()
        node_ids_set = set(node_ids)
        edges_list = [edge for edge in all_edges if edge[0] in node_ids_set]

        if edges_list:
            edges = np.array(edges_list, dtype=np.uint64)
        else:
            edges = np.empty((0, 2), dtype=np.uint64)

        return {
            "image": image,
            "node_ids": node_ids,
            "positions": positions,
            "edges": edges,
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the combined dataset interface.

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - num_timepoints: Number of timepoints (T)
                - volume_shape: Shape of the spatial volume (Z, Y, X)
                - total_nodes: Total number of nodes in the ground truth
                - total_edges: Total number of edges in the ground truth
                - voxel_size: Voxel size (z_size, y_size, x_size) in micrometers
        """
        shape = self.zarr_reader.get_shape()
        stats = self.geff_reader.summary()
        voxel_size = self.zarr_reader.get_voxel_size()

        return {
            "num_timepoints": shape[0],
            "volume_shape": shape[1:],
            "total_nodes": stats["num_nodes"],
            "total_edges": stats["num_edges"],
            "voxel_size": voxel_size,
        }
