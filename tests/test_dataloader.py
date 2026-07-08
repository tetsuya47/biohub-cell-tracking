import pathlib
import pytest
import torch
from torch.utils.data import DataLoader
from src.datasets.dataloader import BiohubDataLoader

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


def test_dataloader_creation() -> None:
    """Test that BiohubDataLoader returns a valid DataLoader."""
    bh_loader = BiohubDataLoader(
        test_zarr_path,
        test_geff_path,
        batch_size=1,
        shuffle=False,
    )
    loader = bh_loader.get_train_loader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 1


def test_dataloader_first_batch() -> None:
    """Test loading first batch and verifying keys, types, shapes, and scaling."""
    bh_loader = BiohubDataLoader(
        test_zarr_path,
        test_geff_path,
        batch_size=1,
        shuffle=False,
    )
    loader = bh_loader.get_train_loader()

    # Load first batch
    batch_iter = iter(loader)
    batch = next(batch_iter)

    assert isinstance(batch, dict)
    expected_keys = {"image", "node_ids", "positions", "edges"}
    assert set(batch.keys()) == expected_keys

    # Check tensor types and batched shape (batch_size = 1 prepends a batch dim)
    image = batch["image"]
    node_ids = batch["node_ids"]
    positions = batch["positions"]
    edges = batch["edges"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(node_ids, torch.Tensor)
    assert isinstance(positions, torch.Tensor)
    assert isinstance(edges, torch.Tensor)

    # Image checks: float32, in range [0, 1]
    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0

    # Shape checks: since batch_size=1, it prepends batch dimension 1
    Z, Y, X = loader.dataset.ds.dataset.zarr_reader.get_shape()[1:]
    assert image.shape == (1, Z, Y, X)

    # Node IDs shape is (1, N_t)
    assert node_ids.ndim == 2
    assert node_ids.shape[0] == 1

    # Positions shape is (1, N_t, 4)
    assert positions.ndim == 3
    assert positions.shape[0] == 1
    assert positions.shape[2] == 4

    # Edges shape is (1, E_t, 2)
    assert edges.ndim == 3
    assert edges.shape[0] == 1
    assert edges.shape[2] == 2
