import pathlib
import pytest
import torch
from src.datasets.torch_dataset import BiohubTorchDataset

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


def test_torch_dataset_loads() -> None:
    """Test that BiohubTorchDataset initializes successfully and binds dataset."""
    dataset = BiohubTorchDataset(test_zarr_path, test_geff_path)
    assert dataset.dataset is not None


def test_torch_dataset_len() -> None:
    """Test len() returns the number of timepoints T."""
    dataset = BiohubTorchDataset(test_zarr_path, test_geff_path)
    T = len(dataset)
    assert T > 0
    assert T == len(dataset.dataset)


def test_torch_dataset_getitem() -> None:
    """Test getitem() returns exact keys, correct torch.Tensor types, and correct shapes."""
    dataset = BiohubTorchDataset(test_zarr_path, test_geff_path)
    T = len(dataset)

    # Load first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    
    # Verify exact keys and nothing else
    expected_keys = {"image", "node_ids", "positions", "edges"}
    assert set(sample.keys()) == expected_keys

    # Verify shapes and torch.Tensor type
    image = sample["image"]
    node_ids = sample["node_ids"]
    positions = sample["positions"]
    edges = sample["edges"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(node_ids, torch.Tensor)
    assert isinstance(positions, torch.Tensor)
    assert isinstance(edges, torch.Tensor)

    # Check alignment of data types and values with the underlying numpy dataset
    np_sample = dataset.dataset[0]
    assert image.dtype == torch.from_numpy(np_sample["image"]).dtype
    assert node_ids.dtype == torch.from_numpy(np_sample["node_ids"]).dtype
    assert positions.dtype == torch.from_numpy(np_sample["positions"]).dtype
    assert edges.dtype == torch.from_numpy(np_sample["edges"]).dtype

    # Check shapes
    assert image.shape == np_sample["image"].shape
    assert node_ids.shape == np_sample["node_ids"].shape
    assert positions.shape == np_sample["positions"].shape
    assert edges.shape == np_sample["edges"].shape

    # Index boundary checks
    with pytest.raises(IndexError):
        dataset[T]
    with pytest.raises(IndexError):
        dataset[-T - 1]
