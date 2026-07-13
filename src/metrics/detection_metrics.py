"""Module for computing baseline detection evaluation metrics."""

import logging
from typing import Dict

import torch

from src.metrics.coordinate_matcher import CoordinateMatcher

logger = logging.getLogger(__name__)


class DetectionMetrics:
    """Class to aggregate and calculate cell detection metrics.

    Calculates Mean Loss, Precision, Recall, and F1 Score by aggregating
    TP, FP, and FN counts computed by CoordinateMatcher over the dataset.
    """

    def __init__(self, distance_threshold: float = 15.0) -> None:
        """Initialize the DetectionMetrics.

        Args:
            distance_threshold: Maximum Euclidean distance for a predicted coordinate
                to match a ground truth coordinate.
        """
        self.distance_threshold = distance_threshold
        self.matcher = CoordinateMatcher(distance_threshold=distance_threshold)
        self.reset()
        logger.info("Initialized DetectionMetrics with distance_threshold=%f", distance_threshold)

    def reset(self) -> None:
        """Reset the metric accumulators."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_loss = 0.0
        self.loss_count = 0
        logger.debug("Metrics accumulators reset.")

    def update(self, predicted_coords: torch.Tensor, gt_coords: torch.Tensor, loss_val: float) -> None:
        """Update metrics state with coordinates and loss from a batch/frame.

        Args:
            predicted_coords: Tensor of shape (M, 3) representing predicted coordinates [z, y, x].
            gt_coords: Tensor of shape (N, 3) representing ground truth coordinates [z, y, x].
            loss_val: The loss value associated with this batch/frame.
        """
        tp, fp, fn = self.matcher.match(predicted_coords, gt_coords)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_loss += loss_val
        self.loss_count += 1
        
        logger.debug(
            "Batch update - TP: %d, FP: %d, FN: %d, Loss: %f. Aggregated TP: %d, FP: %d, FN: %d",
            tp,
            fp,
            fn,
            loss_val,
            self.total_tp,
            self.total_fp,
            self.total_fn,
        )

    def compute(self) -> Dict[str, float]:
        """Compute the final aggregated metrics over the entire dataset.

        Returns:
            Dict[str, float]: Dictionary containing:
                - "mean_loss": The average loss value.
                - "precision": True Positives / (True Positives + False Positives).
                - "recall": True Positives / (True Positives + False Negatives).
                - "f1_score": Harmonic mean of precision and recall.
        """
        mean_loss = self.total_loss / self.loss_count if self.loss_count > 0 else 0.0

        precision = (
            self.total_tp / (self.total_tp + self.total_fp)
            if (self.total_tp + self.total_fp) > 0
            else 0.0
        )
        recall = (
            self.total_tp / (self.total_tp + self.total_fn)
            if (self.total_tp + self.total_fn) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "mean_loss": mean_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
        
        logger.info(
            "Computed final metrics: Loss=%.6f, Precision=%.4f, Recall=%.4f, F1=%.4f",
            mean_loss,
            precision,
            recall,
            f1_score,
        )
        return metrics
