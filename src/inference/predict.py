"""Module for running model inference and peak detection."""

import logging
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

from src.postprocessing.peak_detector import PeakDetector

logger = logging.getLogger(__name__)


def predict(
    model: nn.Module,
    volume: torch.Tensor,
    peak_detector: Optional[PeakDetector] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform inference on a single 3D volume.

    Args:
        model: Baseline detection model.
        volume: Input 3D volume tensor. Can be of shape (Z, Y, X), (1, Z, Y, X), or (1, 1, Z, Y, X).
        peak_detector: PeakDetector instance. If None, instantiates one with threshold 0.5.
        device: Device to run prediction on. If None, uses model's device or CUDA if available.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - prediction_heatmap: Heatmap of shape (1, Z, Y, X).
            - predicted_coordinates: Coordinates of shape (M, 3) representing [z, y, x].
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    if peak_detector is None:
        peak_detector = PeakDetector(threshold=0.5)

    # Prepare input shape to (1, 1, Z, Y, X)
    x = volume.to(device=device, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        # Check if first dim is channel
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Invalid volume shape: {volume.shape}")
    elif x.ndim == 5:
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError(f"Only batch size 1 and channel size 1 supported, got: {volume.shape}")
    else:
        raise ValueError(f"Invalid volume dimension: {volume.ndim}")

    with torch.no_grad():
        prediction = model(x)  # shape (1, 1, Z, Y, X)

    # Extract heatmap shape (1, Z, Y, X)
    prediction_heatmap = prediction[0]

    # Run peak detection
    predicted_coordinates = peak_detector.detect(prediction_heatmap)

    return prediction_heatmap.cpu(), predicted_coordinates.cpu()
