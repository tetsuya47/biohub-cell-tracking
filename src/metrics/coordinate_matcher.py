"""Module for coordinate-based bipartite matching logic."""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


class CoordinateMatcher:
    """Greedy bipartite matcher for 3D coordinates.

    Matches predicted centroids with ground truth annotations based on Euclidean distance
    thresholding.
    """

    def __init__(self, distance_threshold: float = 15.0) -> None:
        """Initialize the CoordinateMatcher.

        Args:
            distance_threshold: Maximum Euclidean distance allowed for a match.
        """
        self.distance_threshold = distance_threshold
        logger.info("Initialized CoordinateMatcher with distance_threshold=%f", distance_threshold)

    def match(self, preds: torch.Tensor, gts: torch.Tensor) -> Tuple[int, int, int]:
        """Match predicted coordinates with ground truth coordinates greedily.

        Args:
            preds: Tensor of shape (M, 3) representing predicted coordinates [z, y, x].
            gts: Tensor of shape (N, 3) representing ground truth coordinates [z, y, x].

        Returns:
            Tuple[int, int, int]: (True Positives, False Positives, False Negatives)
        """
        # Ensure tensors are float and on CPU for safety
        preds = preds.float().cpu()
        gts = gts.float().cpu()

        M = preds.shape[0]
        N = gts.shape[0]

        if M == 0:
            return 0, 0, N
        if N == 0:
            return 0, M, 0

        # Compute Euclidean distance matrix of shape (M, N)
        dists = torch.cdist(preds, gts, p=2)

        tp = 0
        dists_clone = dists.clone()

        # Greedily match the closest points within the threshold
        for _ in range(min(M, N)):
            min_val, min_idx = torch.min(dists_clone.view(-1), dim=0)
            if min_val.item() > self.distance_threshold:
                break
            
            row = min_idx.item() // N
            col = min_idx.item() % N
            tp += 1
            
            # Mask out the matched predicted point and ground truth point
            dists_clone[row, :] = float("inf")
            dists_clone[:, col] = float("inf")

        fp = M - tp
        fn = N - tp
        return tp, fp, fn
