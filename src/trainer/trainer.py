"""Trainer module for training the cell detection baseline model."""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class that encapsulates model, optimizer, scheduler, loss, and training loops."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: The baseline detection PyTorch model.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            loss_fn: Loss function (DetectionLoss).
            lr: Learning rate for the Adam optimizer. Defaults to 1e-3.
            step_size: Epoch step size for StepLR scheduler. Defaults to 10.
            gamma: Multiplicative factor of learning rate decay. Defaults to 0.1.
            device: Device to run training on. If None, auto-detects CUDA/CPU.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        
        # Implement torch.optim.Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Implement torch.optim.lr_scheduler.StepLR
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )
        
        self.epoch = 0
        
        logger.info(
            "Trainer initialized on device '%s' with lr=%s, step_size=%d, gamma=%s",
            self.device,
            lr,
            step_size,
            gamma,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Run a single training step on a batch.

        Args:
            batch: Dictionary containing "image" and "target".

        Returns:
            float: Training loss for this step.
        """
        self.model.train()
        
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)
        
        # If the dataloader does not output a channel dimension, add it.
        # Image shape: (B, Z, Y, X) -> (B, 1, Z, Y, X)
        if images.ndim == 4:
            images = images.unsqueeze(1)
        # Target shape: (B, Z, Y, X) -> (B, 1, Z, Y, X)
        if targets.ndim == 4:
            targets = targets.unsqueeze(1)
            
        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def val_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Run a single validation step on a batch.

        Args:
            batch: Dictionary containing "image" and "target".

        Returns:
            float: Validation loss for this step.
        """
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if targets.ndim == 4:
                targets = targets.unsqueeze(1)
                
            predictions = self.model(images)
            loss = self.loss_fn(predictions, targets)
            
        return loss.item()

    def train_epoch(self) -> float:
        """Run training loop for one epoch.

        Returns:
            float: Average training loss over the epoch.
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss_val = self.train_step(batch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            logger.info(
                "Epoch %d, Iteration %d, Loss: %.6f, LR: %.6e",
                self.epoch,
                batch_idx,
                loss_val,
                current_lr,
            )
            
            total_loss += loss_val
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate_epoch(self) -> float:
        """Run validation loop for one epoch.

        Returns:
            float: Average validation loss over the epoch.
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            loss_val = self.val_step(batch)
            total_loss += loss_val
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info("Epoch %d, Average Validation Loss: %.6f", self.epoch, avg_loss)
        return avg_loss

    def fit(
        self,
        num_epochs: int,
        evaluator: Optional[Any] = None,
        best_checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            evaluator: Optional DetectionEvaluator instance for validation metrics.
            best_checkpoint_path: Optional path to save the best model checkpoint.

        Returns:
            Dict[str, list]: History dictionary containing 'train_loss', 'val_loss',
                'validation_loss', 'precision', 'recall', 'f1_score', and 'learning_rate' lists.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "validation_loss": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "learning_rate": [],
        }

        best_f1 = -1.0
        best_val_loss = float("inf")

        for _ in range(num_epochs):
            self.epoch += 1
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # If evaluator is provided, run automatic evaluation
            if evaluator is not None:
                eval_results = evaluator.evaluate()
                precision = eval_results.get("precision", 0.0)
                recall = eval_results.get("recall", 0.0)
                f1 = eval_results.get("f1_score", 0.0)
                # The evaluator mean_loss corresponds to the validation loss
                eval_val_loss = eval_results.get("mean_loss", val_loss)
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
                eval_val_loss = val_loss

            self.scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["validation_loss"].append(eval_val_loss)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1_score"].append(f1)
            history["learning_rate"].append(current_lr)

            # Check if this is the best epoch and save checkpoint
            if best_checkpoint_path is not None:
                if evaluator is not None:
                    if f1 > best_f1:
                        best_f1 = f1
                        self.save_checkpoint(best_checkpoint_path)
                        logger.info("New best F1 score: %.4f. Saved best checkpoint.", best_f1)
                else:
                    if eval_val_loss < best_val_loss:
                        best_val_loss = eval_val_loss
                        self.save_checkpoint(best_checkpoint_path)
                        logger.info("New best validation loss: %.6f. Saved best checkpoint.", best_val_loss)

        return history

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save a training checkpoint containing model, optimizer, scheduler state and current epoch.

        Args:
            path: Path where the checkpoint will be saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
        }
        
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to '%s' at epoch %d", path, self.epoch)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load a training checkpoint and restore model, optimizer, scheduler state and epoch.

        Args:
            path: Path to the checkpoint file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        
        logger.info("Loaded checkpoint from '%s' starting at epoch %d", path, self.epoch)
