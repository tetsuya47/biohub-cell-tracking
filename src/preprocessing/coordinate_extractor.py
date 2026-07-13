"""Module for extracting spatial coordinates from raw position annotations."""

import logging
import torch

logger = logging.getLogger(__name__)


class GroundTruthCoordinateExtractor:
    """Helper to extract 3D coordinates from ground truth position tensors.

    Isolates evaluation and metrics logic from dataset-specific coordinate layout formats.
    """

    @staticmethod
    def extract(positions: torch.Tensor) -> torch.Tensor:
        """Extract spatial coordinates [z, y, x] of shape (N, 3) from positions.

        Handles positions tensor which has layout [t, z, y, x] or [z, y, x].

        Args:
            positions: Tensor of shape (N, 4) with elements [t, z, y, x],
                or shape (N, 3) with elements [z, y, x].

        Returns:
            torch.Tensor: Coordinates of shape (N, 3) with layout [z, y, x].
        """
        # Ensure we are dealing with a PyTorch Tensor
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions)

        if positions.ndim == 2:
            if positions.shape[-1] == 4:
                # Layout is [t, z, y, x], slice out columns 1, 2, 3
                return positions[:, 1:4]
            elif positions.shape[-1] == 3:
                # Layout is already [z, y, x]
                return positions

        logger.warning(
            "Unexpected positions shape %s. Returning positions as-is.",
            positions.shape,
        )
        return positions
