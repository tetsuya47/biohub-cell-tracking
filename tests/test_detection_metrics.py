"""Unit tests for the CoordinateMatcher and DetectionMetrics classes."""

import pytest
import torch

from src.metrics.coordinate_matcher import CoordinateMatcher
from src.metrics.detection_metrics import DetectionMetrics


def test_matcher_initialization() -> None:
    """Verify that CoordinateMatcher initializes correctly."""
    matcher = CoordinateMatcher(distance_threshold=10.0)
    assert matcher.distance_threshold == 10.0


def test_match_coordinates_exact() -> None:
    """Test greedy coordinate matching on exact matches using CoordinateMatcher."""
    matcher = CoordinateMatcher(distance_threshold=5.0)

    # 3 exact matches
    preds = torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    gts = torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    tp, fp, fn = matcher.match(preds, gts)
    assert tp == 3
    assert fp == 0
    assert fn == 0


def test_match_coordinates_partial() -> None:
    """Test greedy coordinate matching on partial matches using CoordinateMatcher."""
    matcher = CoordinateMatcher(distance_threshold=5.0)

    # Pred 1 to GT 2 is 0.0 (selected first) -> Pred 1 matched to GT 2
    # Pred 2 is [0,0,10], GT 1 is [0,0,3] -> dist 7.0 (exceeds threshold 5.0) -> No match
    # Pred 3 and GT 3 -> dist ~86.6 -> No match
    preds = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [100.0, 100.0, 100.0]])
    gts = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])

    tp, fp, fn = matcher.match(preds, gts)
    assert tp == 1
    assert fp == 2
    assert fn == 2


def test_match_coordinates_empty() -> None:
    """Test CoordinateMatcher behavior when predictions or GT are empty."""
    matcher = CoordinateMatcher(distance_threshold=15.0)

    # Empty predictions
    preds = torch.zeros((0, 3))
    gts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tp, fp, fn = matcher.match(preds, gts)
    assert tp == 0
    assert fp == 0
    assert fn == 2

    # Empty GT
    preds = torch.tensor([[1.0, 2.0, 3.0]])
    gts = torch.zeros((0, 3))
    tp, fp, fn = matcher.match(preds, gts)
    assert tp == 0
    assert fp == 1
    assert fn == 0


def test_metrics_initialization() -> None:
    """Verify that DetectionMetrics initializes with correct default state."""
    metrics = DetectionMetrics(distance_threshold=10.0)
    assert metrics.distance_threshold == 10.0
    assert isinstance(metrics.matcher, CoordinateMatcher)
    assert metrics.total_tp == 0
    assert metrics.total_fp == 0
    assert metrics.total_fn == 0
    assert metrics.total_loss == 0.0
    assert metrics.loss_count == 0


def test_metrics_update_and_compute() -> None:
    """Verify that update accumulates TP/FP/FN/loss correctly and compute outputs correct values."""
    metrics = DetectionMetrics(distance_threshold=5.0)

    # Batch 1: 2 TP, 0 FP, 0 FN, loss = 0.5
    preds_1 = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    gts_1 = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    metrics.update(preds_1, gts_1, loss_val=0.5)

    # Batch 2: 1 TP, 1 FP (at 100,100,100), 1 FN (at 50,50,50), loss = 1.5
    preds_2 = torch.tensor([[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]])
    gts_2 = torch.tensor([[10.0, 10.0, 10.0], [50.0, 50.0, 50.0]])
    metrics.update(preds_2, gts_2, loss_val=1.5)

    assert metrics.total_tp == 3
    assert metrics.total_fp == 1
    assert metrics.total_fn == 1
    assert metrics.total_loss == 2.0
    assert metrics.loss_count == 2

    summary = metrics.compute()

    # Precision = TP / (TP + FP) = 3 / 4 = 0.75
    # Recall = TP / (TP + FN) = 3 / 4 = 0.75
    # F1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
    # Mean loss = 2.0 / 2 = 1.0
    assert summary["mean_loss"] == 1.0
    assert summary["precision"] == 0.75
    assert summary["recall"] == 0.75
    assert summary["f1_score"] == 0.75


def test_metrics_zero_division() -> None:
    """Verify that zero division cases are handled gracefully (return 0.0)."""
    metrics = DetectionMetrics(distance_threshold=5.0)

    # Compute on empty state
    summary = metrics.compute()
    assert summary["mean_loss"] == 0.0
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1_score"] == 0.0
