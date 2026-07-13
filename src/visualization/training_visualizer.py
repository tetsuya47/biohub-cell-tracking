"""Module for training visualization, including loss curves, heatmaps, and peak detections."""

import logging
from pathlib import Path
from typing import Dict, Union, Optional

import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


def plot_loss_curves(
    history: Dict[str, list],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training and validation loss curves over epochs.

    Args:
        history: Dictionary containing training history lists. Must include 'train_loss' and 'val_loss'.
            Can optionally contain 'precision', 'recall', 'f1_score'.
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot losses
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")
    ax1.set_title("Training History")

    # Plot metrics if available on a secondary y-axis
    if "f1_score" in history and any(history["f1_score"]):
        ax2 = ax1.twinx()
        ax2.plot(epochs, history["precision"], "g--", label="Precision")
        ax2.plot(epochs, history["recall"], "y--", label="Recall")
        ax2.plot(epochs, history["f1_score"], "k-", label="F1 Score")
        ax2.set_ylabel("Metrics")
        ax2.tick_params(axis="y")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved loss curves to %s", save_path)
        plt.close()
    else:
        plt.show()


def visualize_prediction_heatmap(
    heatmap: torch.Tensor,
    z_slice: int,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize a 2D slice of the predicted heatmap.

    Args:
        heatmap: Predicted heatmap tensor of shape (1, Z, Y, X) or (Z, Y, X).
        z_slice: The Z-index of the slice to visualize.
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    if heatmap.ndim == 4:
        heatmap_2d = heatmap[0, z_slice].cpu().numpy()
    else:
        heatmap_2d = heatmap[z_slice].cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap_2d, cmap="hot", vmin=0.0, vmax=1.0)
    plt.title(f"Prediction Heatmap (Slice Z={z_slice})")
    plt.colorbar(im)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved prediction heatmap to %s", save_path)
        plt.close()
    else:
        plt.show()


def visualize_target_heatmap(
    heatmap: torch.Tensor,
    z_slice: int,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize a 2D slice of the ground truth target heatmap.

    Args:
        heatmap: Target heatmap tensor of shape (1, Z, Y, X) or (Z, Y, X).
        z_slice: The Z-index of the slice to visualize.
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    if heatmap.ndim == 4:
        heatmap_2d = heatmap[0, z_slice].cpu().numpy()
    else:
        heatmap_2d = heatmap[z_slice].cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap_2d, cmap="hot", vmin=0.0, vmax=1.0)
    plt.title(f"Target Heatmap (Slice Z={z_slice})")
    plt.colorbar(im)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved target heatmap to %s", save_path)
        plt.close()
    else:
        plt.show()


def visualize_peak_detection(
    heatmap: torch.Tensor,
    coordinates: torch.Tensor,
    z_slice: int,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize detected cell peaks overlaid on the heatmap.

    Args:
        heatmap: Prediction or target heatmap tensor of shape (1, Z, Y, X) or (Z, Y, X).
        coordinates: Detected coordinates tensor of shape (M, 3) representing [z, y, x].
        z_slice: The Z-index of the slice to visualize.
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    if heatmap.ndim == 4:
        heatmap_2d = heatmap[0, z_slice].cpu().numpy()
    else:
        heatmap_2d = heatmap[z_slice].cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap_2d, cmap="hot", vmin=0.0, vmax=1.0)
    plt.colorbar(im)

    # Filter coordinates falling within this slice
    if coordinates.numel() > 0:
        slice_mask = torch.round(coordinates[:, 0]).long() == z_slice
        slice_coords = coordinates[slice_mask]
        if slice_coords.shape[0] > 0:
            plt.scatter(
                slice_coords[:, 2].cpu().numpy(),
                slice_coords[:, 1].cpu().numpy(),
                color="cyan",
                marker="x",
                s=100,
                linewidths=2,
                label="Detected Peaks",
            )
            plt.legend()

    plt.title(f"Peak Detection (Slice Z={z_slice})")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved peak detection to %s", save_path)
        plt.close()
    else:
        plt.show()


def compare_prediction_vs_target(
    prediction_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    z_slice: int,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Compare prediction vs target side-by-side.

    Args:
        prediction_heatmap: Predicted heatmap tensor.
        target_heatmap: Ground truth target heatmap tensor.
        z_slice: The Z-index of the slice to visualize.
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    if prediction_heatmap.ndim == 4:
        pred_2d = prediction_heatmap[0, z_slice].cpu().numpy()
    else:
        pred_2d = prediction_heatmap[z_slice].cpu().numpy()

    if target_heatmap.ndim == 4:
        target_2d = target_heatmap[0, z_slice].cpu().numpy()
    else:
        target_2d = target_heatmap[z_slice].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    im1 = axes[0].imshow(pred_2d, cmap="hot", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Prediction Heatmap (Z={z_slice})")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(target_2d, cmap="hot", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Target Heatmap (Z={z_slice})")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved prediction vs target comparison to %s", save_path)
        plt.close()
    else:
        plt.show()


def visualize_detected_coordinates_3d(
    detected_coords: torch.Tensor,
    gt_coords: Optional[torch.Tensor] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize detected and ground truth coordinates in 3D.

    Args:
        detected_coords: Detected coordinates tensor of shape (M, 3) representing [z, y, x].
        gt_coords: Optional ground truth coordinates tensor of shape (N, 3) representing [z, y, x].
        save_path: Path to save the generated figure. If None, figure is shown.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    det = detected_coords.cpu().numpy()
    if det.shape[0] > 0:
        ax.scatter(det[:, 2], det[:, 1], det[:, 0], c="r", marker="x", s=50, label="Detected")

    if gt_coords is not None:
        gt = gt_coords.cpu().numpy()
        if gt.shape[0] > 0:
            ax.scatter(
                gt[:, 2],
                gt[:, 1],
                gt[:, 0],
                c="g",
                marker="o",
                s=30,
                label="Ground Truth",
                alpha=0.5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Coordinates Comparison")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info("Saved 3D coordinates comparison to %s", save_path)
        plt.close()
    else:
        plt.show()
