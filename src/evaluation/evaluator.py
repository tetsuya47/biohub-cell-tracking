"""Module for orchestrating validation evaluation of detection models."""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.postprocessing.peak_detector import PeakDetector
from src.metrics.detection_metrics import DetectionMetrics
from src.preprocessing.coordinate_extractor import GroundTruthCoordinateExtractor

logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Orchestrates the evaluation workflow for baseline cell detection.

    Runs model inference, post-processing peak detection, and coordinate-based
    metrics computation over a validation dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        peak_detector: PeakDetector,
        metrics: DetectionMetrics,
        loss_fn: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the DetectionEvaluator.

        Args:
            model: PyTorch baseline detection model.
            val_loader: DataLoader for the validation set.
            peak_detector: Post-processing PeakDetector instance.
            metrics: DetectionMetrics instance to track and compute results.
            loss_fn: Loss function (criterion) to compute model loss.
            device: Device to run evaluation on. If None, auto-detects.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.val_loader = val_loader
        self.peak_detector = peak_detector
        self.metrics = metrics
        self.loss_fn = loss_fn.to(self.device)

        logger.info(
            "DetectionEvaluator initialized on device '%s' with %d validation batches.",
            self.device,
            len(val_loader) if hasattr(val_loader, "__len__") else -1,
        )

    def evaluate(self) -> Dict[str, float]:
        """Run the evaluation loop over the validation loader.

        Switches the model to evaluation mode, disables gradient computation,
        and aggregates detection metrics.

        Returns:
            Dict[str, float]: Compiled evaluation summary containing:
                - "mean_loss": Mean validation loss.
                - "precision": Cell detection precision score.
                - "recall": Cell detection recall score.
                - "f1_score": Cell detection F1 score.
        """
        self.model.eval()
        self.metrics.reset()

        logger.info("Starting detection evaluation...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                # TODO: Image/target channel consistency must be guaranteed by the data pipeline in a future refactor.

                # Model forward pass
                predictions = self.model(images)

                # Compute step loss
                loss = self.loss_fn(predictions, targets)
                loss_val = loss.item()

                # Process batch predictions
                batch_size = images.shape[0]
                for b in range(batch_size):
                    # Peak detector expects heatmap shape (1, Z, Y, X) or (B, 1, Z, Y, X)
                    # We pass a slice of shape (1, Z, Y, X) representing a single frame
                    pred_heatmap = predictions[b]  # shape (1, Z, Y, X)
                    pred_coords = self.peak_detector.detect(pred_heatmap)

                    # Extract ground truth positions
                    pos_b = batch["positions"][b]

                    # Extract 3D coordinates [z, y, x] using the preprocessing utility
                    gt_coords = GroundTruthCoordinateExtractor.extract(pos_b)

                    # Update metrics state
                    self.metrics.update(pred_coords, gt_coords, loss_val)

                if (batch_idx + 1) % 10 == 0:
                    logger.info("Processed %d validation batches.", batch_idx + 1)

        summary = self.metrics.compute()
        logger.info("Evaluation complete. Results: %s", summary)
        return summary
