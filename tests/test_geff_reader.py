import pathlib
import pytest
import numpy as np
from src.datasets.geff_reader import GEFFReader

# Find the first .geff dataset in the train directory
TRAIN_DIR = pathlib.Path("data/raw/train")
GEFF_PATHS = sorted(list(TRAIN_DIR.glob("*.geff")))

if not GEFF_PATHS:
    pytest.skip("No GEFF datasets found in data/raw/train", allow_module_level=True)

TEST_GEFF_PATH = GEFF_PATHS[0]


def test_reader_not_loaded_raises_exception() -> None:
    """Test that calling getter methods before loading raises ValueError."""
    reader = GEFFReader()
    with pytest.raises(ValueError, match="Dataset is not loaded"):
        reader.get_node_ids()
    with pytest.raises(ValueError, match="Dataset is not loaded"):
        reader.get_edges()
    with pytest.raises(ValueError, match="Dataset is not loaded"):
        reader.get_node_positions()
    with pytest.raises(ValueError, match="Dataset is not loaded"):
        reader.summary()


def test_reader_loads_dataset() -> None:
    """Test that reader successfully opens geff dataset and has path attribute."""
    reader = GEFFReader(TEST_GEFF_PATH)
    assert reader.path == TEST_GEFF_PATH.resolve()
    assert reader._store is not None


def test_get_node_ids() -> None:
    """Test node ID extraction shape and data type."""
    reader = GEFFReader(TEST_GEFF_PATH)
    node_ids = reader.get_node_ids()
    assert isinstance(node_ids, np.ndarray)
    assert node_ids.ndim == 1
    assert node_ids.dtype == np.uint64
    assert len(node_ids) > 0


def test_get_edges() -> None:
    """Test directed edges extraction shape and data type."""
    reader = GEFFReader(TEST_GEFF_PATH)
    edges = reader.get_edges()
    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert edges.dtype == np.uint64


def test_get_node_positions() -> None:
    """Test spatial-temporal coordinates extraction shape and alignment."""
    reader = GEFFReader(TEST_GEFF_PATH)
    node_ids = reader.get_node_ids()
    positions = reader.get_node_positions()
    assert isinstance(positions, np.ndarray)
    assert positions.shape == (len(node_ids), 4)
    # Check dimensions (t, z, y, x)
    for i in range(4):
        # Coordinates should be integers/floats
        assert np.issubdtype(positions[:, i].dtype, np.integer) or np.issubdtype(positions[:, i].dtype, np.floating)


def test_summary() -> None:
    """Test summary statistics dict fields and values consistency."""
    reader = GEFFReader(TEST_GEFF_PATH)
    stats = reader.summary()
    assert isinstance(stats, dict)
    for key in ["num_nodes", "num_edges", "t_range", "z_range", "y_range", "x_range"]:
        assert key in stats

    assert stats["num_nodes"] == len(reader.get_node_ids())
    assert stats["num_edges"] == len(reader.get_edges())

    # Range values verification
    for range_key in ["t_range", "z_range", "y_range", "x_range"]:
        val_min, val_max = stats[range_key]
        assert val_min <= val_max
