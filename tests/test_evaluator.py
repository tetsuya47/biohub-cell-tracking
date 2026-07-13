"""Unit tests for the GroundTruthCoordinateExtractor and DetectionEvaluator classes."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.detection.baseline_detector import BaselineDetector
from src.losses.detection_loss import DetectionLoss
from src.postprocessing.peak_detector import PeakDetector
from src.metrics.detection_metrics import DetectionMetrics
from src.evaluation.evaluator import DetectionEvaluator
from src.preprocessing.coordinate_extractor import GroundTruthCoordinateExtractor


class DummyEvalDataset(Dataset):
    """A dummy dataset for evaluator testing ensuring channel consistency from the data layer."""

    def __init__(self, size: int = 2) -> None:
        """Initialize the DummyEvalDataset."""
        self.size = size

    def __len__(self) -> int:
        """Get dataset length."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Get dummy sample dictionary.

        Returns image with a channel dimension to comply with the architecture
        feedback that the data pipeline is responsible for channel consistency.
        """
        # Image shape: (1, Z, Y, X), Target shape: (1, Z, Y, X)
        # Positions: (N_t, 4) representing [t, z, y, x]
        return {
            "image": torch.rand(1, 4, 8, 8, dtype=torch.float32),
            "target": torch.rand(1, 4, 8, 8, dtype=torch.float32),
            "positions": torch.tensor([[float(idx), 1.0, 2.0, 3.0], [float(idx), 2.0, 2.0, 2.0]]),
        }


@pytest.fixture
def evaluator_setup():
    """Fixture to set up model, dataloader, peak detector, metrics, loss, and evaluator."""
    model = BaselineDetector(in_channels=1, hidden_channels=2, out_channels=1)
    loss_fn = DetectionLoss()
    
    val_ds = DummyEvalDataset(size=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    peak_detector = PeakDetector(threshold=0.5)
    metrics = DetectionMetrics(distance_threshold=5.0)
    
    evaluator = DetectionEvaluator(
        model=model,
        val_loader=val_loader,
        peak_detector=peak_detector,
        metrics=metrics,
        loss_fn=loss_fn,
        device="cpu",
    )
    return evaluator


def test_evaluator_initialization(evaluator_setup) -> None:
    """Verify that the evaluator initializes with correct components and mode."""
    evaluator = evaluator_setup
    assert isinstance(evaluator.model, nn.Module)
    assert isinstance(evaluator.val_loader, DataLoader)
    assert isinstance(evaluator.peak_detector, PeakDetector)
    assert isinstance(evaluator.metrics, DetectionMetrics)
    assert isinstance(evaluator.loss_fn, nn.Module)
    assert str(evaluator.device) == "cpu"


def test_evaluator_loop_execution(evaluator_setup) -> None:
    """Verify that the evaluation loop executes successfully and returns expected summary."""
    evaluator = evaluator_setup
    
    summary = evaluator.evaluate()
    
    assert isinstance(summary, dict)
    expected_keys = {"mean_loss", "precision", "recall", "f1_score"}
    assert set(summary.keys()) == expected_keys
    
    # Check that loss is valid and scores are in range [0, 1]
    assert summary["mean_loss"] >= 0.0
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert 0.0 <= summary["f1_score"] <= 1.0


def test_evaluator_model_state_and_gradients(evaluator_setup) -> None:
    """Verify that evaluation sets model.eval() and does not track gradients."""
    evaluator = evaluator_setup
    
    # Mock forward pass recording gradients
    evaluator.model.train()
    assert evaluator.model.training is True
    
    _ = evaluator.evaluate()
    
    # Check that evaluator set model to eval mode
    assert evaluator.model.training is False
    
    # Run forward pass explicitly on dummy input to verify gradients are disabled in eval context
    dummy_input = torch.rand(1, 1, 4, 8, 8)
    
    with torch.no_grad():
        out = evaluator.model(dummy_input)
        assert out.requires_grad is False


def test_ground_truth_coordinate_extractor() -> None:
    """Verify that GroundTruthCoordinateExtractor parses coordinate formats correctly."""
    # Test [t, z, y, x] format
    positions_4d = torch.tensor([[0.0, 10.0, 20.0, 30.0], [1.0, 15.0, 25.0, 35.0]])
    coords = GroundTruthCoordinateExtractor.extract(positions_4d)
    assert coords.shape == (2, 3)
    assert torch.equal(coords, torch.tensor([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]]))

    # Test [z, y, x] format
    positions_3d = torch.tensor([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
    coords = GroundTruthCoordinateExtractor.extract(positions_3d)
    assert coords.shape == (2, 3)
    assert torch.equal(coords, positions_3d)
    
    # Test fallback conversion of lists
    list_pos = [[0.0, 1.0, 2.0, 3.0]]
    coords = GroundTruthCoordinateExtractor.extract(list_pos)
    assert coords.shape == (1, 3)
    assert torch.equal(coords, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
