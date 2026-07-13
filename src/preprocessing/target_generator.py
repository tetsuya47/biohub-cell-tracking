"""Module for converting GEFF annotations (node coordinates) into dense training targets."""

import logging
import torch

from src.preprocessing.coordinate_extractor import GroundTruthCoordinateExtractor

logger = logging.getLogger(__name__)


class TargetGenerator:
    """Ground Truth Target Generator.

    Converts sparse cell node coordinates from GEFF annotations into dense
    supervision targets (e.g., 3D heatmaps) for training.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int] = (8, 32, 32),
        sigma: float | None = None,
        radius: float | None = None,
    ) -> None:
        """Initialize the TargetGenerator.

        Args:
            volume_shape: Spatial shape of the output heatmap volume (Z, Y, X).
            sigma: Standard deviation of the Gaussian kernel. If None, loaded from config.
            radius: Radius within which the Gaussian is computed. If None, loaded from config.
        """
        self.volume_shape = volume_shape

        # Load from configuration if not explicitly provided
        if sigma is None or radius is None:
            try:
                from src.utils.config import load_hydra_config
                cfg = load_hydra_config()
                if sigma is None:
                    sigma = float(cfg.dataset.get("sigma", 1.5))
                if radius is None:
                    radius = float(cfg.dataset.get("radius", 3.0))
            except Exception as e:
                logger.warning(
                    "Could not load config for TargetGenerator, using defaults (sigma=1.5, radius=3.0): %s",
                    e,
                )
                if sigma is None:
                    sigma = 1.5
                if radius is None:
                    radius = 3.0

        self.sigma = sigma
        self.radius = radius
        logger.info(
            "Initializing TargetGenerator with volume_shape=%s, sigma=%s, radius=%s",
            volume_shape,
            self.sigma,
            self.radius,
        )

    def generate(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate a dense supervision heatmap target from node positions.

        Args:
            positions: Tensor of shape (N, 4) containing coordinate points [t, z, y, x],
                or shape (N, 3) containing [z, y, x].

        Returns:
            torch.Tensor: Generated target heatmap of shape (1, Z, Y, X).
        """
        Z, Y, X = self.volume_shape
        target = torch.zeros((1, Z, Y, X), dtype=torch.float32, device=positions.device)

        if positions.numel() == 0:
            logger.debug("Empty positions tensor. Returning zero heatmap.")
            return target

        # Extract coordinates of layout [z, y, x]
        coords = GroundTruthCoordinateExtractor.extract(positions)

        for coord in coords:
            z0, y0, x0 = coord[0], coord[1], coord[2]

            # Define local bounding box cropped to volume boundaries
            z_min = max(0, int(torch.floor(z0 - self.radius).item()))
            z_max = min(Z - 1, int(torch.ceil(z0 + self.radius).item()))
            y_min = max(0, int(torch.floor(y0 - self.radius).item()))
            y_max = min(Y - 1, int(torch.ceil(y0 + self.radius).item()))
            x_min = max(0, int(torch.floor(x0 - self.radius).item()))
            x_max = min(X - 1, int(torch.ceil(x0 + self.radius).item()))

            # Skip if coordinate is completely outside the volume boundaries
            if z_min > z_max or y_min > y_max or x_min > x_max:
                continue

            # Create local coordinate grids on matching device
            z_range = torch.arange(z_min, z_max + 1, dtype=positions.dtype, device=positions.device)
            y_range = torch.arange(y_min, y_max + 1, dtype=positions.dtype, device=positions.device)
            x_range = torch.arange(x_min, x_max + 1, dtype=positions.dtype, device=positions.device)

            Z_grid, Y_grid, X_grid = torch.meshgrid(z_range, y_range, x_range, indexing="ij")

            # Compute squared distances
            dist_sq = (Z_grid - z0) ** 2 + (Y_grid - y0) ** 2 + (X_grid - x0) ** 2

            # Compute Gaussian values
            g = torch.exp(-dist_sq / (2.0 * self.sigma * self.sigma))

            # Mask out voxels beyond the configured radius
            g[dist_sq > self.radius * self.radius] = 0.0

            # Merge overlapping Gaussians using maximum value
            target[0, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = torch.maximum(
                target[0, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1],
                g,
            )

        # Normalize output to [0, 1] by clamping
        target = torch.clamp(target, 0.0, 1.0)
        logger.debug("Generated target heatmap shape: %s", target.shape)
        return target
