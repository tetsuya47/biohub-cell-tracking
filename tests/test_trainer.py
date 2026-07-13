"""Unit tests for the Trainer class."""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.detection.baseline_detector import BaselineDetector
from src.losses.detection_loss import DetectionLoss
from src.trainer.trainer import Trainer


class DummyDataset(Dataset):
    """A dummy dataset for Trainer unit testing."""

    def __init__(self, size: int = 4) -> None:
        """Initialize the DummyDataset.

        Args:
            size: Number of samples in the dummy dataset.
        """
        self.size = size

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Get a dummy sample dictionary."""
        # BaselineDetector expects images of shape (B, 1, Z, Y, X) and targets of shape (B, 1, Z, Y, X)
        # DataLoader adds batch dimension B, so we return image (Z, Y, X) and target (1, Z, Y, X)
        return {
            "image": torch.rand(8, 16, 16, dtype=torch.float32),
            "target": torch.rand(1, 8, 16, 16, dtype=torch.float32),
        }


@pytest.fixture
def trainer_setup():
    """Fixture to set up a model, dataloaders, loss, and trainer."""
    model = BaselineDetector(in_channels=1, hidden_channels=4, out_channels=1)
    loss_fn = DetectionLoss()
    
    train_ds = DummyDataset(size=4)
    val_ds = DummyDataset(size=2)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        lr=1e-3,
        step_size=5,
        gamma=0.5,
        device="cpu",
    )
    return trainer


def test_trainer_initialization(trainer_setup) -> None:
    """Verify that the Trainer initializes model, optimizer, and scheduler correctly."""
    trainer = trainer_setup
    assert isinstance(trainer.model, nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
    assert isinstance(trainer.loss_fn, nn.Module)
    assert trainer.epoch == 0
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3


def test_trainer_steps(trainer_setup) -> None:
    """Verify that a single training step and a single validation step execute successfully."""
    trainer = trainer_setup
    batch = next(iter(trainer.train_loader))
    
    # Run a training step
    train_loss = trainer.train_step(batch)
    assert isinstance(train_loss, float)
    assert train_loss >= 0.0
    
    # Run a validation step
    val_loss = trainer.val_step(batch)
    assert isinstance(val_loss, float)
    assert val_loss >= 0.0


def test_trainer_epochs(trainer_setup) -> None:
    """Verify that full training and validation epoch methods run successfully."""
    trainer = trainer_setup
    
    train_loss = trainer.train_epoch()
    assert isinstance(train_loss, float)
    assert train_loss >= 0.0
    
    val_loss = trainer.validate_epoch()
    assert isinstance(val_loss, float)
    assert val_loss >= 0.0


def test_trainer_fit_and_scheduler_step(trainer_setup) -> None:
    """Verify that the fit function updates the training epoch and steps the scheduler."""
    trainer = trainer_setup
    
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    
    # step_size is 5. If we run for 5 epochs, the LR should step.
    history = trainer.fit(num_epochs=5)
    
    assert trainer.epoch == 5
    assert len(history["train_loss"]) == 5
    assert len(history["val_loss"]) == 5
    
    final_lr = trainer.optimizer.param_groups[0]["lr"]
    assert final_lr == initial_lr * 0.5


def test_checkpoint_save_and_load(trainer_setup) -> None:
    """Verify that checkpoint saving and loading works correctly and restores model state."""
    trainer = trainer_setup
    
    # Save current model weights to verify after loading
    initial_weights = [p.clone() for p in trainer.model.parameters()]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        
        # Save checkpoint
        trainer.epoch = 42
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
        
        # Modify model parameters to mock training changes
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(1.0)
                
        # Modify trainer state
        trainer.epoch = 0
        
        # Load checkpoint and assert weights are restored
        trainer.load_checkpoint(checkpoint_path)
        
        assert trainer.epoch == 42
        for p_initial, p_restored in zip(initial_weights, trainer.model.parameters()):
            assert torch.allclose(p_initial, p_restored)
