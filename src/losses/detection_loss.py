"""Loss function module for cell detection."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DetectionLoss(nn.Module):
    """Mean Squared Error (MSE) loss for baseline cell detection."""

    def __init__(self) -> None:
        """Initialize the DetectionLoss."""
        super().__init__()
        logger.info("Initializing DetectionLoss (using nn.MSELoss)")
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the MSE loss between predictions and targets.

        Args:
            predictions: Predicted detection map of shape (B, C, Z, Y, X).
            targets: Target ground truth map of shape (B, C, Z, Y, X).

        Returns:
            torch.Tensor: Mean squared error loss scalar tensor.
        """
        loss = self.mse(predictions, targets)
        logger.debug("Computed detection loss: %f", float(loss.item()))
        return loss
