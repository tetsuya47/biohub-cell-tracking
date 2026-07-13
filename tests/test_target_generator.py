"""Unit tests for TargetGenerator class."""

import pytest
import torch

from src.preprocessing.target_generator import TargetGenerator


def test_target_generator_shape() -> None:
    """Verify target generator output shape for different inputs."""
    generator = TargetGenerator(volume_shape=(8, 16, 16), sigma=1.5, radius=3.0)

    # Check shape for N,4 input
    positions_4 = torch.tensor([[0.0, 2.0, 5.0, 5.0]])
    target_4 = generator.generate(positions_4)
    assert target_4.shape == (1, 8, 16, 16)

    # Check shape for N,3 input
    positions_3 = torch.tensor([[2.0, 5.0, 5.0]])
    target_3 = generator.generate(positions_3)
    assert target_3.shape == (1, 8, 16, 16)

    # Empty positions should return zero heatmap of correct shape
    empty_positions = torch.zeros((0, 3))
    target_empty = generator.generate(empty_positions)
    assert target_empty.shape == (1, 8, 16, 16)
    assert torch.all(target_empty == 0.0)


def test_target_generator_single_point() -> None:
    """Verify Gaussian generation for a single point."""
    generator = TargetGenerator(volume_shape=(10, 10, 10), sigma=1.0, radius=2.0)
    # Point at center (5.0, 5.0, 5.0)
    positions = torch.tensor([[5.0, 5.0, 5.0]])
    target = generator.generate(positions)

    # Center value should be exactly 1.0
    assert torch.allclose(target[0, 5, 5, 5], torch.tensor(1.0))

    # Check that distance-based falloff is correct
    # Distance = 1.0 voxel: exp(-1.0^2 / 2) = exp(-0.5) ≈ 0.6065
    assert torch.allclose(target[0, 6, 5, 5], torch.tensor(0.6065), atol=1e-4)

    # Distance = 2.0 voxels: exp(-2.0^2 / 2) = exp(-2.0) ≈ 0.1353
    assert torch.allclose(target[0, 7, 5, 5], torch.tensor(0.1353), atol=1e-4)

    # Distance > 2.0 voxels: should be exactly 0.0 due to radius mask
    assert target[0, 8, 5, 5] == 0.0


def test_target_generator_multiple_points_and_overlapping() -> None:
    """Verify that multiple points and overlapping Gaussians merge using maximum value."""
    generator = TargetGenerator(volume_shape=(10, 10, 10), sigma=1.0, radius=2.0)
    # Two points, they will overlap at (5, 5, 5)
    positions = torch.tensor([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]])

    # Generate heatmap for combined points
    target_combined = generator.generate(positions)

    # Generate heatmaps for individual points
    target_1 = generator.generate(positions[0:1])
    target_2 = generator.generate(positions[1:2])

    # Combined target should be the element-wise maximum of individual targets
    target_max = torch.maximum(target_1, target_2)
    assert torch.allclose(target_combined, target_max)


def test_target_generator_border_cases() -> None:
    """Verify target generator behavior near volume boundaries and out of bounds."""
    generator = TargetGenerator(volume_shape=(10, 10, 10), sigma=1.0, radius=2.0)

    # Point at the corner (0.0, 0.0, 0.0)
    positions_corner = torch.tensor([[0.0, 0.0, 0.0]])
    target_corner = generator.generate(positions_corner)
    assert target_corner[0, 0, 0, 0] == 1.0
    # Values should be clipped and not cause errors
    assert target_corner[0, 3, 0, 0] == 0.0  # outside radius

    # Point completely outside the volume (e.g. at -5.0 or 15.0)
    positions_outside = torch.tensor([[-5.0, 15.0, 5.0]])
    target_outside = generator.generate(positions_outside)
    assert torch.all(target_outside == 0.0)  # should be completely zero and not crash


def test_target_generator_output_range() -> None:
    """Verify that output values are strictly normalized to [0, 1]."""
    generator = TargetGenerator(volume_shape=(10, 10, 10), sigma=1.0, radius=3.0)

    # Check with multiple overlapping points
    positions = torch.tensor([[5.0, 5.0, 5.0], [5.2, 5.0, 5.0], [4.8, 5.0, 5.0]])
    target = generator.generate(positions)

    # Values should be in [0, 1] range
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)


def test_target_generator_config_loading() -> None:
    """Verify that configuration parameters are correctly loaded."""
    # Since config has sigma=1.5 and radius=3.0
    generator = TargetGenerator(volume_shape=(10, 10, 10))
    assert generator.sigma == 1.5
    assert generator.radius == 3.0
