"""Configuration utilities for the project using Hydra and OmegaConf."""

import logging
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)


def load_hydra_config(config_name: str = "config", config_dir: str = "../configs") -> DictConfig:
    """Load configuration parameters using Hydra.

    Args:
        config_name: The name of the root config file.
        config_dir: Directory containing configuration files, relative to this module or absolute.

    Returns:
        DictConfig: Loaded OmegaConf configuration dictionary.
    """
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_dir)
    
    if not config_path.is_absolute():
        # Try relative to this file
        abs_path = (Path(__file__).parent / config_path).resolve()
        if not abs_path.exists():
            # Try relative to project root
            abs_path = (project_root / config_path).resolve()
        if not abs_path.exists():
            # Fall back to root-level configs
            abs_path = (project_root / "configs").resolve()
    else:
        abs_path = config_path

    logger.info("Loading config %s from absolute path %s", config_name, abs_path)

    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass

    with hydra.initialize_config_dir(config_dir=str(abs_path), version_base="1.1"):
        cfg = hydra.compose(config_name=config_name)

    return cfg
