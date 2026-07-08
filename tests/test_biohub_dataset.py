import pathlib
import pytest
import numpy as np
from src.datasets.biohub_dataset import BiohubDataset

# Find first dataset pair programmatically
TRAIN_DIR = pathlib.Path("data/raw/train")
ZARR_PATHS = sorted(list(TRAIN_DIR.glob("*.zarr")))
GEFF_PATHS = sorted(list(TRAIN_DIR.glob("*.geff")))

if not ZARR_PATHS or not GEFF_PATHS:
    pytest.skip("No dataset pairs found in data/raw/train", allow_module_level=True)

# Select first pair that matches base name
test_zarr_path = None
test_geff_path = None

for z_path in ZARR_PATHS:
    g_path = z_path.with_suffix(".geff")
    if g_path in GEFF_PATHS:
        test_zarr_path = z_path
        test_geff_path = g_path
        break

if test_zarr_path is None or test_geff_path is None:
    pytest.skip("No matching Zarr/GEFF pair found", allow_module_level=True)


def test_biohub_dataset_loads() -> None:
    """Test that BiohubDataset initializes successfully and binds readers."""
    dataset = BiohubDataset(test_zarr_path, test_geff_path)
    assert dataset.zarr_reader is not None
    assert dataset.geff_reader is not None


def test_biohub_dataset_len() -> None:
    """Test len() returns the number of timepoints T."""
    dataset = BiohubDataset(test_zarr_path, test_geff_path)
    T = len(dataset)
    assert T > 0
    assert T == dataset.zarr_reader.get_shape()[0]


def test_biohub_dataset_getitem() -> None:
    """Test getitem() returns exact keys, correct shapes, and handles boundary checks."""
    dataset = BiohubDataset(test_zarr_path, test_geff_path)
    T = len(dataset)

    # Load first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    
    # Verify exact keys and nothing else
    expected_keys = {"image", "node_ids", "positions", "edges"}
    assert set(sample.keys()) == expected_keys

    # Verify shapes and types
    image = sample["image"]
    node_ids = sample["node_ids"]
    positions = sample["positions"]
    edges = sample["edges"]

    Z, Y, X = dataset.zarr_reader.get_shape()[1:]
    assert image.shape == (Z, Y, X)
    assert node_ids.ndim == 1
    assert node_ids.dtype == np.uint64
    assert positions.ndim == 2
    assert positions.shape[1] == 4  # [t, z, y, x]
    assert edges.ndim == 2
    assert edges.shape[1] == 2  # [source, target]
    assert edges.dtype == np.uint64

    # Index boundary checks
    with pytest.raises(IndexError):
        dataset[T]
    with pytest.raises(IndexError):
        dataset[-T - 1]


def test_biohub_dataset_summary() -> None:
    """Test summary() dictionary content consistency."""
    dataset = BiohubDataset(test_zarr_path, test_geff_path)
    s = dataset.summary()
    assert isinstance(s, dict)
    
    expected_keys = {"num_timepoints", "volume_shape", "total_nodes", "total_edges", "voxel_size"}
    assert set(s.keys()) == expected_keys

    assert s["num_timepoints"] == len(dataset)
    assert s["volume_shape"] == dataset.zarr_reader.get_shape()[1:]
    assert s["total_nodes"] == len(dataset.geff_reader.get_node_ids())
    assert s["total_edges"] == len(dataset.geff_reader.get_edges())
    assert len(s["voxel_size"]) == 3
