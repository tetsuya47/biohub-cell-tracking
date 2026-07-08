import pathlib
import pytest
import numpy as np
from src.datasets.zarr_dataset import ZarrDatasetReader, DatasetNotOpenError

# Find the first .zarr dataset in the train directory
TRAIN_DIR = pathlib.Path("data/raw/train")
ZARR_PATHS = sorted(list(TRAIN_DIR.glob("*.zarr")))

if not ZARR_PATHS:
    pytest.skip("No Zarr datasets found in data/raw/train", allow_module_level=True)

TEST_DATASET_PATH = ZARR_PATHS[0]


def test_reader_not_open_raises_exception() -> None:
    """Test that calling getter methods before opening raises DatasetNotOpenError."""
    reader = ZarrDatasetReader()
    with pytest.raises(DatasetNotOpenError):
        reader.get_shape()
    with pytest.raises(DatasetNotOpenError):
        reader.get_dtype()
    with pytest.raises(DatasetNotOpenError):
        reader.get_voxel_size()
    with pytest.raises(DatasetNotOpenError):
        reader.get_statistics()
    with pytest.raises(DatasetNotOpenError):
        reader.get_frame(0)
    with pytest.raises(DatasetNotOpenError):
        reader.get_slice_xy(0, 0)
    with pytest.raises(DatasetNotOpenError):
        reader.get_slice_xz(0, 0)
    with pytest.raises(DatasetNotOpenError):
        reader.get_slice_yz(0, 0)


def test_reader_loads_dataset() -> None:
    """Test that reader successfully opens dataset and matches expected structure."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    assert reader.path == TEST_DATASET_PATH.resolve()
    assert reader._array is not None


def test_get_shape_and_dtype() -> None:
    """Test shape and dtype property retrieval."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    shape = reader.get_shape()
    assert len(shape) == 4
    assert shape[0] > 0  # T
    assert shape[1] > 0  # Z
    assert shape[2] > 0  # Y
    assert shape[3] > 0  # X

    dtype = reader.get_dtype()
    assert isinstance(dtype, np.dtype)


def test_get_voxel_size() -> None:
    """Test voxel size retrieval from metadata."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    voxel_size = reader.get_voxel_size()
    assert len(voxel_size) == 3
    # Check that scales are positive numbers
    for scale in voxel_size:
        assert isinstance(scale, float)
        assert scale > 0.0


def test_get_frame() -> None:
    """Test frame retrieval and boundary checks."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    shape = reader.get_shape()
    T, Z, Y, X = shape

    # Test valid frame load (first and last frame)
    frame_first = reader.get_frame(0)
    assert frame_first.shape == (Z, Y, X)
    assert frame_first.dtype == reader.get_dtype()

    frame_last = reader.get_frame(-1)
    assert frame_last.shape == (Z, Y, X)

    # Test out of bounds index
    with pytest.raises(IndexError):
        reader.get_frame(T)
    with pytest.raises(IndexError):
        reader.get_frame(-T - 1)


def test_slice_extraction() -> None:
    """Test slice extraction in all three orthogonal views (XY, XZ, YZ)."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    T, Z, Y, X = reader.get_shape()

    # XY Slice
    slice_xy = reader.get_slice_xy(0, 0)
    assert slice_xy.shape == (Y, X)

    # XZ Slice
    slice_xz = reader.get_slice_xz(0, 0)
    assert slice_xz.shape == (Z, X)

    # YZ Slice
    slice_yz = reader.get_slice_yz(0, 0)
    assert slice_yz.shape == (Z, Y)

    # Boundary checks
    with pytest.raises(IndexError):
        reader.get_slice_xy(T, 0)
    with pytest.raises(IndexError):
        reader.get_slice_xy(0, Z)
    with pytest.raises(IndexError):
        reader.get_slice_xz(0, Y)
    with pytest.raises(IndexError):
        reader.get_slice_yz(0, X)


def test_statistics() -> None:
    """Test that statistics returns a dict with min, max, mean, std."""
    reader = ZarrDatasetReader(TEST_DATASET_PATH)
    # We test on first timepoint array to verify fast stats calculation
    stats = reader.get_statistics()
    assert isinstance(stats, dict)
    for key in ["min", "max", "mean", "std"]:
        assert key in stats
        assert isinstance(stats[key], (int, float, np.integer, np.floating))

    assert stats["min"] <= stats["max"]
    assert stats["std"] >= 0.0
