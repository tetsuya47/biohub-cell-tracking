"""Module for data pre-processing and scaling transforms."""

from typing import Dict

import torch


class TransformPipeline:
    """Pre-processing transform pipeline for Biohub Cell Tracking data.

    Converts the image tensor to float32 and scales it to [0.0, 1.0], leaving
    ground truth lineage targets (node_ids, positions, edges) unchanged.
    """

    def __init__(self) -> None:
        """Initialize the TransformPipeline."""
        pass

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply pre-processing transforms to a sample.

        Args:
            sample: Dictionary containing exactly:
                - "image": torch.Tensor of shape (Z, Y, X).
                - "node_ids": torch.Tensor of shape (N_t,).
                - "positions": torch.Tensor of shape (N_t, 4).
                - "edges": torch.Tensor of shape (E_t, 2).

        Returns:
            Dict[str, torch.Tensor]: The transformed sample with scaled image.
        """
        image = sample["image"].to(torch.float32) / 65535.0

        return {
            "image": image,
            "node_ids": sample["node_ids"],
            "positions": sample["positions"],
            "edges": sample["edges"],
        }
