"""Unit tests for the prediction module."""

import pytest
import torch

from src.models.detection.baseline_detector import BaselineDetector
from src.inference.predict import predict


def test_predict_executes() -> None:
    """Verify that inference executes and returns correct shapes and types."""
    model = BaselineDetector(in_channels=1, hidden_channels=4, out_channels=1)

    # Create input volume of shape (Z, Y, X)
    volume = torch.rand(8, 16, 16)

    heatmap, coords = predict(model=model, volume=volume)

    # Verify types
    assert isinstance(heatmap, torch.Tensor)
    assert isinstance(coords, torch.Tensor)

    # Verify shape
    # The output heatmap shape should be (1, Z, Y, X)
    assert heatmap.shape == (1, 8, 16, 16)

    # Verify coords type and dimensions
    assert coords.ndim == 2
    assert coords.shape[1] == 3  # [z, y, x]
