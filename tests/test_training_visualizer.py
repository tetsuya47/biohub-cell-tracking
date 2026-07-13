"""Unit tests for the training visualizer module."""

import tempfile
from pathlib import Path
import pytest
import torch

from src.visualization.training_visualizer import (
    plot_loss_curves,
    visualize_prediction_heatmap,
    visualize_target_heatmap,
    visualize_peak_detection,
    compare_prediction_vs_target,
    visualize_detected_coordinates_3d,
)


def test_visualizers_execute() -> None:
    """Verify that all visualization functions execute without errors and save figures."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Test plot_loss_curves
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.45, 0.35, 0.25],
            "precision": [0.5, 0.6, 0.7],
            "recall": [0.4, 0.5, 0.6],
            "f1_score": [0.44, 0.54, 0.64],
        }
        loss_curve_path = temp_path / "loss_curves.png"
        plot_loss_curves(history, save_path=loss_curve_path)
        assert loss_curve_path.exists()

        # 2. Test heatmaps and overlays
        pred_heatmap = torch.rand(1, 8, 16, 16)
        target_heatmap = torch.rand(1, 8, 16, 16)
        coords = torch.tensor([[4.0, 5.0, 5.0], [4.0, 10.0, 10.0]])

        pred_path = temp_path / "pred_heatmap.png"
        visualize_prediction_heatmap(pred_heatmap, z_slice=4, save_path=pred_path)
        assert pred_path.exists()

        target_path = temp_path / "target_heatmap.png"
        visualize_target_heatmap(target_heatmap, z_slice=4, save_path=target_path)
        assert target_path.exists()

        peak_path = temp_path / "peak_detection.png"
        visualize_peak_detection(pred_heatmap, coords, z_slice=4, save_path=peak_path)
        assert peak_path.exists()

        comp_path = temp_path / "comparison.png"
        compare_prediction_vs_target(
            pred_heatmap, target_heatmap, z_slice=4, save_path=comp_path
        )
        assert comp_path.exists()

        # 3. Test 3D visualization
        coords_3d_path = temp_path / "coords_3d.png"
        visualize_detected_coordinates_3d(coords, coords, save_path=coords_3d_path)
        assert coords_3d_path.exists()
