"""Module for reading cell tracking datasets stored in Zarr v3 format."""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import numpy as np
import zarr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatasetNotOpenError(Exception):
    """Raised when performing operations on a dataset that has not been opened."""

    pass


class ZarrDatasetReader:
    """Reader for cell tracking datasets stored in Zarr v3 format.

    Provides high-level access to 3D volumes and orthogonal slices (XY, XZ, YZ)
    along with dataset metadata (voxel size, dtype, shape, and statistics).
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the reader.

        Optionally opens the dataset if path is provided.

        Args:
            path: Path to the Zarr dataset directory.
        """
        self.path: Optional[Path] = None
        self._store: Optional[zarr.Group] = None
        self._array: Optional[zarr.Array] = None
        self.dataset_name: str = "0"  # Default array path within the group

        if path is not None:
            self.open_dataset(path)

    def open_dataset(self, path: Union[str, Path]) -> None:
        """Open the Zarr dataset and load metadata.

        Args:
            path: Path to the Zarr dataset directory.

        Raises:
            FileNotFoundError: If the directory or zarr.json does not exist.
            ValueError: If the dataset is not a valid Zarr v3 structure.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")

        # In Zarr v3, we open the root directory as a group
        logging.info(f"Opening Zarr dataset group at: {self.path}")
        try:
            self._store = zarr.open(str(self.path))
        except Exception as e:
            raise ValueError(f"Failed to open Zarr store at {self.path}: {e}")

        # Check if the store is a Group
        if not isinstance(self._store, zarr.Group):
            raise ValueError(f"Zarr store at {self.path} is not a Group.")

        # Read zarr.json in root to check dataset layout
        zarr_json_path = self.path / "zarr.json"
        if not zarr_json_path.exists():
            raise FileNotFoundError(f"zarr.json metadata file not found at {zarr_json_path}")

        try:
            with open(zarr_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Determine array path (defaults to "0")
            multiscales = metadata.get("attributes", {}).get("multiscales", [])
            if multiscales:
                datasets = multiscales[0].get("datasets", [])
                if datasets:
                    self.dataset_name = datasets[0].get("path", "0")
        except Exception as e:
            logging.warning(f"Could not parse zarr.json metadata: {e}. Falling back to default path '0'.")
            self.dataset_name = "0"

        if self.dataset_name not in self._store:
            raise ValueError(f"Array '{self.dataset_name}' not found in Zarr group {self.path}")

        self._array = self._store[self.dataset_name]
        if not isinstance(self._array, zarr.Array):
            raise ValueError(f"Member '{self.dataset_name}' of Zarr store is not an Array.")

        logging.info(
            f"Successfully opened dataset '{self.dataset_name}' with shape {self._array.shape} "
            f"and dtype {self._array.dtype}"
        )

    def get_shape(self) -> Tuple[int, int, int, int]:
        """Get the shape of the dataset as (T, Z, Y, X).

        Returns:
            Tuple[int, int, int, int]: Shape of the dataset.

        Raises:
            DatasetNotOpenError: If the dataset is not open.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")
        
        shape = self._array.shape
        if len(shape) != 4:
            raise ValueError(f"Expected a 4D array (T, Z, Y, X), but got shape {shape}")
        
        return tuple(shape)  # type: ignore

    def get_dtype(self) -> np.dtype:
        """Get the numpy data type of the array.

        Returns:
            np.dtype: Data type of the array elements.

        Raises:
            DatasetNotOpenError: If the dataset is not open.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")
        return self._array.dtype

    def get_voxel_size(self) -> Tuple[float, float, float]:
        """Read voxel spatial size from zarr.json metadata.

        Parses the coordinate transformations for the spatial dimensions.

        Returns:
            Tuple[float, float, float]: Voxel size as (z_size, y_size, x_size) in micrometers.

        Raises:
            DatasetNotOpenError: If the dataset is not open.
            FileNotFoundError: If zarr.json does not exist.
            ValueError: If coordinate transformations scale could not be found.
        """
        if self.path is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        zarr_json_path = self.path / "zarr.json"
        if not zarr_json_path.exists():
            raise FileNotFoundError(f"zarr.json metadata file not found at {zarr_json_path}")

        with open(zarr_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        try:
            multiscales = metadata["attributes"]["multiscales"]
            axes = multiscales[0]["axes"]
            datasets = multiscales[0]["datasets"]

            scale_transform = None
            for ds in datasets:
                if ds["path"] == self.dataset_name:
                    for transform in ds.get("coordinateTransformations", []):
                        if transform["type"] == "scale":
                            scale_transform = transform["scale"]
                            break
            
            if scale_transform is None:
                for ds in datasets:
                    for transform in ds.get("coordinateTransformations", []):
                        if transform["type"] == "scale":
                            scale_transform = transform["scale"]
                            break

            if scale_transform is None:
                raise ValueError("Scale coordinate transformation not found in metadata.")

            # Filter scale values matching space axes
            space_scales = []
            for idx, axis in enumerate(axes):
                if axis.get("type") == "space":
                    space_scales.append(scale_transform[idx])

            if len(space_scales) != 3:
                logging.warning(
                    f"Expected 3 space axes, found {len(space_scales)}. "
                    f"Falling back to last 3 elements of scale transform {scale_transform}."
                )
                if len(scale_transform) >= 3:
                    return tuple(scale_transform[-3:])  # type: ignore
                return tuple(scale_transform)  # type: ignore

            return tuple(space_scales)  # type: ignore

        except Exception as e:
            logging.error(f"Error parsing voxel size from zarr.json: {e}")
            raise ValueError(f"Failed to parse voxel size from metadata: {e}")

    def get_statistics(self) -> Dict[str, float]:
        """Compute dataset statistics (min, max, mean, std) across all timepoints.

        Calculates statistics incrementally to avoid memory overflow when handling
        large arrays.

        Returns:
            Dict[str, float]: A dictionary containing 'min', 'max', 'mean', and 'std'.

        Raises:
            DatasetNotOpenError: If the dataset is not open.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        arr = self._array
        shape = arr.shape
        T = shape[0]

        val_min = float("inf")
        val_max = float("-inf")
        sum_val = 0.0
        sum_sq = 0.0
        N = 0

        logging.info(f"Computing statistics incrementally for {self.path} (T={T})...")
        for t in range(T):
            frame = arr[t]
            val_min = min(val_min, float(frame.min()))
            val_max = max(val_max, float(frame.max()))
            sum_val += float(frame.sum())
            sum_sq += float(np.sum(frame.astype(np.float64) ** 2))
            N += frame.size

        mean = sum_val / N
        variance = max(0.0, (sum_sq / N) - (mean ** 2))
        std = np.sqrt(variance)

        return {
            "min": val_min,
            "max": val_max,
            "mean": mean,
            "std": std,
        }

    def get_frame(self, t: int) -> np.ndarray:
        """Get 3D volume at a specific timepoint.

        Args:
            t: Timepoint index. Supports negative indexing.

        Returns:
            np.ndarray: 3D volume of shape (Z, Y, X).

        Raises:
            DatasetNotOpenError: If the dataset is not open.
            IndexError: If the timepoint index is out of bounds.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        shape = self.get_shape()
        T = shape[0]

        # Normalize indices
        if t < 0:
            t += T

        if t < 0 or t >= T:
            raise IndexError(f"Timepoint index {t} out of bounds for dataset of shape {shape}")

        return np.asarray(self._array[t])

    def get_slice_xy(self, t: int, z: int) -> np.ndarray:
        """Get XY slice at timepoint t and depth z.

        Args:
            t: Timepoint index. Supports negative indexing.
            z: Depth index. Supports negative indexing.

        Returns:
            np.ndarray: XY image slice of shape (Y, X).

        Raises:
            DatasetNotOpenError: If the dataset is not open.
            IndexError: If indices are out of bounds.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        shape = self.get_shape()
        T, Z, Y, X = shape

        if t < 0:
            t += T
        if z < 0:
            z += Z

        if t < 0 or t >= T or z < 0 or z >= Z:
            raise IndexError(f"Indices t={t}, z={z} out of bounds for dataset of shape {shape}")

        return np.asarray(self._array[t, z, :, :])

    def get_slice_xz(self, t: int, y: int) -> np.ndarray:
        """Get XZ slice at timepoint t and vertical position y.

        Args:
            t: Timepoint index. Supports negative indexing.
            y: Vertical position index. Supports negative indexing.

        Returns:
            np.ndarray: XZ image slice of shape (Z, X).

        Raises:
            DatasetNotOpenError: If the dataset is not open.
            IndexError: If indices are out of bounds.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        shape = self.get_shape()
        T, Z, Y, X = shape

        if t < 0:
            t += T
        if y < 0:
            y += Y

        if t < 0 or t >= T or y < 0 or y >= Y:
            raise IndexError(f"Indices t={t}, y={y} out of bounds for dataset of shape {shape}")

        return np.asarray(self._array[t, :, y, :])

    def get_slice_yz(self, t: int, x: int) -> np.ndarray:
        """Get YZ slice at timepoint t and horizontal position x.

        Args:
            t: Timepoint index. Supports negative indexing.
            x: Horizontal position index. Supports negative indexing.

        Returns:
            np.ndarray: YZ image slice of shape (Z, Y).

        Raises:
            DatasetNotOpenError: If the dataset is not open.
            IndexError: If indices are out of bounds.
        """
        if self._array is None:
            raise DatasetNotOpenError("Dataset is not open. Call open_dataset() first.")

        shape = self.get_shape()
        T, Z, Y, X = shape

        if t < 0:
            t += T
        if x < 0:
            x += X

        if t < 0 or t >= T or x < 0 or x >= X:
            raise IndexError(f"Indices t={t}, x={x} out of bounds for dataset of shape {shape}")

        return np.asarray(self._array[t, :, :, x])
