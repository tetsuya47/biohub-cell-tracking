import pytest
from omegaconf import DictConfig
from src.utils.config import load_hydra_config

def test_load_hydra_config() -> None:
    """
    Verifies that the Hydra configuration system correctly loads default parameters.
    """
    # Note: relative path is '../../configs' from src/utils/config.py
    # Since we are running from tests, the relative path config_dir will resolve correctly.
    cfg = load_hydra_config(config_name="config", config_dir="../configs")
    
    assert isinstance(cfg, DictConfig)
    assert "dataset" in cfg
    assert "model" in cfg
    assert "train" in cfg
    assert "tracking" in cfg
    assert "logging" in cfg

    # Verify key dataset parameters
    assert cfg.dataset.name == "biohub_cell_tracking"
    assert len(cfg.dataset.volume_shape) == 4
    
    # Verify tracking hyperparameters
    assert cfg.tracking.search_radius > 0.0
    assert cfg.tracking.link_threshold > 0.0
