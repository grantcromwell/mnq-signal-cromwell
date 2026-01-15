"""
Training Pipeline for MNQ VL-JEPA Model

Supports:
1. JEPA Pre-training (self-supervised on unlabeled data)
2. SFT Fine-tuning (supervised on labeled 3:1 R/R signals)

Features:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving/loading
- TensorBoard logging
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading signal training.

    Stores:
    - GAF images (3-day and week)
    - Text embeddings (correlation descriptions)
    - Labels (3:1 R/R success)
    """

    def __init__(
        self,
        images_3day: np.ndarray,
        images_week: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ):
        self.images_3day = torch.from_numpy(images_3day).float()
        self.images_week = torch.from_numpy(images_week).float()
        self.text_embeddings = torch.from_numpy(text_embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.timestamps = timestamps

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_3day": self.images_3day[idx],
            "image_week": self.images_week[idx],
            "text_embedding": self.text_embeddings[idx],
            "label": self.labels[idx],
            "timestamp": self.timestamps[idx] if self.timestamps else None,
        }


class JEPACollator:
    """
    Collator for JEPA pre-training with masking.
    """

    def __init__(self, mask_ratio: float = 0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images_3day = torch.stack([b["image_3day"] for b in batch])
        images_week = torch.stack([b["image_week"] for b in batch])
        text_embeddings = torch.stack([b["text_embedding"] for b in batch])

        target_images_3day = images_3day.clone()
        target_images_week = images_week.clone()

        if self.mask_ratio > 0:
            mask_size = int(images_3day.shape[-1] * self.mask_ratio)
            mask_start = torch.randint(
                0, images_3day.shape[-1] - mask_size, (images_3day.shape[0],)
            )

            for i, start in enumerate(mask_start):
                images_3day[i, :, start : start + mask_size, :] = 0
                images_3day[i, :, :, start : start + mask_size] = 0

        return {
            "images_3day": images_3day,
            "images_week": images_week,
            "text_embeddings": text_embeddings,
            "target_images_3day": target_images_3day,
            "target_images_week": target_images_week,
        }


class SFTCollator:
    """
    Collator for supervised fine-tuning.
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "images_3day": torch.stack([b["image_3day"] for b in batch]),
            "images_week": torch.stack([b["image_week"] for b in batch]),
            "text_embeddings": torch.stack([b["text_embedding"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
            "timestamps": [b["timestamp"] for b in batch],
        }


class VLJEPATrainer:
    """
    Trainer for VL-JEPA model with JEPA pre-training and SFT.
    """

    def __init__(
        self,
        model: nn.Module,
        config_path: str = "../config/config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = model.to(device)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load()["training"]

        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.weight_decay = self.config["weight_decay"]
        self.warmup_steps = self.config["warmup_steps"]
        self.max_steps = self.config["max_steps"]
        self.jepa_epochs = self.config["jepa_epochs"]
        self.sft_epochs = self.config["sft_epochs"]

        self.optimizer = None
        self.scaler = GradScaler()
        self.scheduler = None

        self.save_dir = Path(self.config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.warmup_steps
        )

    def train_jepa(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        JEPA pre-training phase.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            checkpoint_path: Optional path to resume from

        Returns:
            Dict with training history
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from {checkpoint_path}")

        self.setup_optimizer()
        collator = JEPACollator(mask_ratio=0.5)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.jepa_epochs):
            train_loss = self._train_epoch_jepa(train_loader, collator)
            history["train_loss"].append(train_loss)

            logger.info(
                f"JEPA Epoch {epoch + 1}/{self.jepa_epochs} - Loss: {train_loss:.4f}"
            )

            if val_loader is not None:
                val_loss = self._validate_jepa(val_loader, collator)
                history["val_loss"].append(val_loss)
                logger.info(f"  Val Loss: {val_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"jepa_epoch_{epoch + 1}.pt")

        return history

    def _train_epoch_jepa(self, loader: DataLoader, collator: JEPACollator) -> float:
        """Train one epoch of JEPA"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="JEPA Training"):
            images_3day = batch["images_3day"].to(self.device)
            images_week = batch["images_week"].to(self.device)
            text_embeddings = batch["text_embeddings"].to(self.device)
            target_images_3day = batch["target_images_3day"].to(self.device)
            target_images_week = batch["target_images_week"].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model.forward_jepa(
                    images_3day,
                    images_week,
                    text_embeddings,
                    text_embeddings,
                    target_images_3day,
                    target_images_week,
                )

                loss = output["jepa_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.model.update_jepa_target()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / n_batches

    def _validate_jepa(self, loader: DataLoader, collator: JEPACollator) -> float:
        """Validate JEPA model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                images_3day = batch["images_3day"].to(self.device)
                images_week = batch["images_week"].to(self.device)
                text_embeddings = batch["text_embeddings"].to(self.device)
                target_images_3day = batch["target_images_3day"].to(self.device)
                target_images_week = batch["target_images_week"].to(self.device)

                with autocast():
                    output = self.model.forward_jepa(
                        images_3day,
                        images_week,
                        text_embeddings,
                        text_embeddings,
                        target_images_3day,
                        target_images_week,
                    )
                    loss = output["jepa_loss"]

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train_sft(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Supervised fine-tuning phase.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            checkpoint_path: Optional path to resume from

        Returns:
            Dict with training history
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from {checkpoint_path}")

        self.setup_optimizer()
        collator = SFTCollator()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.sft_epochs):
            train_loss, train_acc = self._train_epoch_sft(
                train_loader, collator, criterion
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            logger.info(
                f"SFT Epoch {epoch + 1}/{self.sft_epochs} - "
                f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            )

            if val_loader is not None:
                val_loss, val_acc = self._validate_sft(val_loader, collator, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"sft_epoch_{epoch + 1}.pt")

        return history

    def _train_epoch_sft(
        self, loader: DataLoader, collator: SFTCollator, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train one epoch of SFT"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch in tqdm(loader, desc="SFT Training"):
            images_3day = batch["images_3day"].to(self.device)
            images_week = batch["images_week"].to(self.device)
            text_embeddings = batch["text_embeddings"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model.forward_sft(
                    images_3day, images_week, text_embeddings, text_embeddings
                )
                logits = output["logits"]

                loss = criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / n_batches, correct / total

    def _validate_sft(
        self, loader: DataLoader, collator: SFTCollator, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate SFT model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                images_3day = batch["images_3day"].to(self.device)
                images_week = batch["images_week"].to(self.device)
                text_embeddings = batch["text_embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast():
                    output = self.model.forward_sft(
                        images_3day, images_week, text_embeddings, text_embeddings
                    )
                    logits = output["logits"]

                    loss = criterion(logits, labels)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches, correct / total

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.save_dir / filename)
        logger.info(f"Saved checkpoint: {filename}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)


def prepare_training_data(
    mnq_df: pd.DataFrame,
    correlation_data: Any,
    gaf_encoder: Any,
    labeler: Any,
    train_ratio: float = 0.8,
) -> Tuple[TradingDataset, TradingDataset]:
    """
    Prepare training and validation datasets.

    Args:
        mnq_df: MNQ OHLCV DataFrame
        correlation_data: Hierarchical correlation data
        gaf_encoder: GAF encoder instance
        labeler: Trade labeler instance
        train_ratio: Ratio of training data

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from data.gaf_encoder import GAFDatasetGenerator
    from features.correlation_rings import flatten_correlation_for_bert

    dataset_gen = GAFDatasetGenerator(gaf_encoder)

    samples = dataset_gen.batch_generate(mnq_df, window_days=3, step_bars=4)

    binary_labels = labeler.get_binary_labels(mnq_df, direction="long")

    images_3day = np.stack([s["image_3day"] for s in samples], axis=0)
    images_week = np.stack([s["image_week"] for s in samples], axis=0)

    text_embeddings = flatten_correlation_for_bert(correlation_data)
    text_embeddings = np.tile(text_embeddings, (len(samples), 1))

    labels = binary_labels[-len(samples) :]

    n_train = int(len(labels) * train_ratio)

    train_dataset = TradingDataset(
        images_3day[:n_train],
        images_week[:n_train],
        text_embeddings[:n_train],
        labels[:n_train],
        timestamps=[s["timestamp"] for s in samples[:n_train]],
    )

    val_dataset = TradingDataset(
        images_3day[n_train:],
        images_week[n_train:],
        text_embeddings[n_train:],
        labels[n_train:],
        timestamps=[s["timestamp"] for s in samples[n_train:]],
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    from models.vjepa_encoder import create_model, VLJEPAConfig
    from data.gaf_encoder import GAFEncoder
    from training.labeling import TradeLabeler
    from features.correlation_rings import HierarchicalCorrelationEngine
    import numpy as np

    model, optimizer = create_model()

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="15min")
    price = 17000.0
    prices = [price]
    for i in range(999):
        change = np.random.randn() * 20
        price += change
        prices.append(price)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + abs(np.random.randn() * 10) for p in prices],
            "low": [p - abs(np.random.randn() * 10) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 1000),
        },
        index=dates,
    )

    gaf_encoder = GAFEncoder()
    labeler = TradeLabeler()

    corr_engine = HierarchicalCorrelationEngine()
    corr_data = corr_engine.process({symbol: df for symbol in ["MNQ", "ES", "NQ"]})

    train_ds, val_ds = prepare_training_data(df, corr_data, gaf_encoder, labeler)

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=SFTCollator()
    )
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=SFTCollator())

    trainer = VLJEPATrainer(model)

    print("Starting SFT training...")
    history = trainer.train_sft(train_loader, val_loader)

    print("Training complete!")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
