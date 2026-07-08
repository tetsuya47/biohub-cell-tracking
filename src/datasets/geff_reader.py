"""Module for reading cell lineage ground truth tracking files in .geff format (Zarr v3)."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import zarr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GEFFReader:
    """Reader for Biohub Cell Tracking ground truth datasets stored in .geff format.

    Provides access to nodes, edges, node spatial-temporal coordinates, and summary
    statistics of the lineage graph.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the GEFFReader.

        Optionally loads the dataset if a path is provided.

        Args:
            path: Path to the .geff directory.
        """
        self.path: Optional[Path] = None
        self._store: Optional[zarr.Group] = None

        if path is not None:
            self.load(path)

    def load(self, path: Union[str, Path]) -> None:
        """Load/open the Zarr group representing the .geff dataset.

        Args:
            path: Path to the .geff directory.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the dataset does not conform to the expected format.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        logging.info(f"Loading .geff dataset at: {self.path}")
        try:
            self._store = zarr.open(str(self.path))
        except Exception as e:
            raise ValueError(f"Failed to open Zarr store at {self.path}: {e}")

        # Basic validation of required paths in .geff
        required_paths = [
            "nodes/ids",
            "edges/ids",
            "nodes/props/t/values",
            "nodes/props/z/values",
            "nodes/props/y/values",
            "nodes/props/x/values",
        ]
        for rp in required_paths:
            if rp not in self._store:
                raise ValueError(f"Required Zarr member '{rp}' is missing in {self.path}")

        logging.info(
            f"Successfully loaded .geff dataset with {self._store['nodes/ids'].shape[0]} nodes "
            f"and {self._store['edges/ids'].shape[0]} edges."
        )

    def get_node_ids(self) -> np.ndarray:
        """Get all node IDs.

        Returns:
            np.ndarray: 1D array of node IDs (uint64).

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._store is None:
            raise ValueError("Dataset is not loaded. Call load() first.")
        return np.asarray(self._store["nodes/ids"])

    def get_edges(self) -> np.ndarray:
        """Get all directed edges.

        Each edge is represented by a pair of node IDs [source_id, target_id].

        Returns:
            np.ndarray: 2D array of shape (E, 2) containing source and target node IDs (uint64).

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._store is None:
            raise ValueError("Dataset is not loaded. Call load() first.")
        return np.asarray(self._store["edges/ids"])

    def get_node_positions(self) -> np.ndarray:
        """Get spatial-temporal positions (t, z, y, x) for all nodes.

        Returns:
            np.ndarray: 2D array of shape (N, 4) containing [t, z, y, x] positions in
                the same order as node IDs returned by get_node_ids().

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._store is None:
            raise ValueError("Dataset is not loaded. Call load() first.")

        t = np.asarray(self._store["nodes/props/t/values"])
        z = np.asarray(self._store["nodes/props/z/values"])
        y = np.asarray(self._store["nodes/props/y/values"])
        x = np.asarray(self._store["nodes/props/x/values"])

        return np.stack([t, z, y, x], axis=1)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset.

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - num_nodes: Number of nodes
                - num_edges: Number of edges
                - t_range: Tuple of (min_t, max_t)
                - z_range: Tuple of (min_z, max_z)
                - y_range: Tuple of (min_y, max_y)
                - x_range: Tuple of (min_x, max_x)

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._store is None:
            raise ValueError("Dataset is not loaded. Call load() first.")

        node_ids = self.get_node_ids()
        edges = self.get_edges()
        positions = self.get_node_positions()

        num_nodes = len(node_ids)
        num_edges = len(edges)

        if num_nodes > 0:
            t_min, t_max = int(positions[:, 0].min()), int(positions[:, 0].max())
            z_min, z_max = int(positions[:, 1].min()), int(positions[:, 1].max())
            y_min, y_max = int(positions[:, 2].min()), int(positions[:, 2].max())
            x_min, x_max = int(positions[:, 3].min()), int(positions[:, 3].max())
        else:
            t_min, t_max = 0, 0
            z_min, z_max = 0, 0
            y_min, y_max = 0, 0
            x_min, x_max = 0, 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "t_range": (t_min, t_max),
            "z_range": (z_min, z_max),
            "y_range": (y_min, y_max),
            "x_range": (x_min, x_max),
        }
