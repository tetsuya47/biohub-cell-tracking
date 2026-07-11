"""Unit tests for the baseline detection model, detection loss, target generator, and peak detector."""

import pytest
import torch
from src.models.detection.baseline_detector import BaselineDetector
from src.losses.detection_loss import DetectionLoss
from src.preprocessing.target_generator import TargetGenerator
from src.postprocessing.peak_detector import PeakDetector


def test_model_initializes() -> None:
    """Verify that the BaselineDetector model initializes correctly."""
    model = BaselineDetector()
    assert isinstance(model, torch.nn.Module)


def test_forward_pass_works_and_shape_is_correct() -> None:
    """Verify that the forward pass works and produces the expected output shape."""
    model = BaselineDetector(in_channels=1, hidden_channels=16, out_channels=1)

    # Input shape: (B, C, Z, Y, X)
    batch_size = 2
    in_channels = 1
    depth = 8
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, depth, height, width)
    output = model(x)

    # Expected output shape: (B, 1, Z, Y, X)
    expected_shape = (batch_size, 1, depth, height, width)
    assert output.shape == expected_shape

    # Sigmoid should constrain output to [0.0, 1.0]
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_loss_computes() -> None:
    """Verify that the DetectionLoss computes the loss correctly."""
    loss_fn = DetectionLoss()
    assert isinstance(loss_fn, torch.nn.Module)

    # Input shape: (B, C, Z, Y, X)
    shape = (2, 1, 8, 32, 32)
    predictions = torch.rand(shape)
    targets = torch.rand(shape)

    loss = loss_fn(predictions, targets)

    # Expected scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_target_generator() -> None:
    """Verify that the TargetGenerator generates correct shapes and does not fail."""
    generator = TargetGenerator(volume_shape=(8, 32, 32))
    positions = torch.randn(5, 4)
    target = generator.generate(positions)

    assert isinstance(target, torch.Tensor)
    assert target.shape == (1, 8, 32, 32)


def test_peak_detector() -> None:
    """Verify that the PeakDetector extracts peak coordinates from heatmaps."""
    detector = PeakDetector(threshold=0.5)
    heatmap = torch.rand(2, 1, 8, 32, 32)
    coords = detector.detect(heatmap)

    assert isinstance(coords, torch.Tensor)
    assert coords.ndim == 2
    assert coords.shape[1] == 3
