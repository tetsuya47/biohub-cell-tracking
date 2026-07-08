import pathlib
import pytest
import torch
from src.datasets.torch_dataset import BiohubTorchDataset
from src.preprocessing.transforms import TransformPipeline

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


def test_transform_pipeline_execution() -> None:
    """Test that TransformPipeline runs and transforms image tensor properly."""
    dataset = BiohubTorchDataset(test_zarr_path, test_geff_path)
    pipeline = TransformPipeline()

    raw_sample = dataset[0]
    transformed_sample = pipeline(raw_sample)

    assert isinstance(transformed_sample, dict)
    assert set(transformed_sample.keys()) == {"image", "node_ids", "positions", "edges"}

    # Check image changes
    image_raw = raw_sample["image"]
    image_transformed = transformed_sample["image"]

    assert image_transformed.dtype == torch.float32
    assert image_transformed.min() >= 0.0
    assert image_transformed.max() <= 1.0

    # Ensure correct scaling calculation
    expected_scaled = image_raw.to(torch.float32) / 65535.0
    assert torch.allclose(image_transformed, expected_scaled)

    # Check that targets are unchanged (exact reference/equality check)
    assert torch.equal(transformed_sample["node_ids"], raw_sample["node_ids"])
    assert torch.equal(transformed_sample["positions"], raw_sample["positions"])
    assert torch.equal(transformed_sample["edges"], raw_sample["edges"])
