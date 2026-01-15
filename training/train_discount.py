"""
Discount Detection Training Pipeline

Trains the discount detector model on labeled MNQ data.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.discount_labeler import DiscountLabeler, LabelingConfig
from models.discount_detector import DiscountDetector, DiscountModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscountTrainer:
    """
    Training pipeline for discount detection model.
    """

    def __init__(
        self,
        model: DiscountDetector,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.class_weights = (
            class_weights.to(device) if class_weights is not None else None
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        label_smoothing: float = 0.1,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            label_smoothing: Label smoothing factor

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(batch_x)

            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    label_smoothing=label_smoothing,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            loss = loss_fn(output["logits"], batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(
        self,
        val_loader: DataLoader,
        confidence_threshold: float = 0.50,
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            confidence_threshold: Confidence threshold for YES signals

        Returns:
            Dict with metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)

                if self.class_weights is not None:
                    loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output["logits"], batch_y)
                total_loss += loss.item()

                preds = (output["confidence"] >= confidence_threshold).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_confidences.extend(output["confidence"].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

        return {
            "val_loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "avg_confidence": np.mean(all_confidences),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        label_smoothing: float = 0.1,
        confidence_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            patience: Early stopping patience
            label_smoothing: Label smoothing factor
            confidence_threshold: Confidence threshold for YES signals

        Returns:
            Dict with training history
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        history = {
            "train_loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, label_smoothing)
            val_metrics = self.validate(val_loader, confidence_threshold)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics["val_loss"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["precision"].append(val_metrics["precision"])
            history["recall"].append(val_metrics["recall"])
            history["f1"].append(val_metrics["f1"])
            history["accuracy"].append(val_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Prec: {val_metrics['precision']:.3f} - "
                f"Recall: {val_metrics['recall']:.3f} - "
                f"F1: {val_metrics['f1']:.3f}"
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        final_metrics = self.validate(val_loader, confidence_threshold)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
        }


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_seed: int = 42,
    use_yfinance: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Load data and prepare dataloaders.

    Args:
        data_path: Path to OHLCV data CSV
        test_size: Fraction for test set
        batch_size: Batch size
        random_seed: Random seed
        use_yfinance: Data is from yfinance format

    Returns:
        Tuple of (train_loader, val_loader, data_info)
    """
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    if use_yfinance:
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    labeler = DiscountLabeler()
    feature_df, X, y = labeler.create_labeled_dataset(df)

    valid_mask = y != -1
    X = X[valid_mask]
    y = y[valid_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    X_val = (X_val - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_dist = np.bincount(y_train.astype(int))
    failures = int(class_dist[0])
    successes = int(class_dist[1]) if len(class_dist) > 1 else 0

    class_weights = np.array(
        [
            failures / (failures + successes + 1e-8),
            successes / (failures + successes + 1e-8),
        ]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    data_info = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "num_features": X_train.shape[1],
        "class_distribution": {
            "failures": failures,
            "successes": successes,
        },
        "class_weights": class_weights.tolist(),
    }

    logger.info(f"Training samples: {data_info['train_size']}")
    logger.info(f"Validation samples: {data_info['val_size']}")
    logger.info(f"Class distribution: {data_info['class_distribution']}")

    return train_loader, val_loader, data_info


def train_discount_model(
    data_path: str,
    output_dir: str = "results/discount",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 15,
    confidence_threshold: float = 0.80,
    use_yfinance: bool = False,
) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        data_path: Path to OHLCV data
        output_dir: Output directory for results
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        confidence_threshold: Confidence threshold for YES signals
        use_yfinance: Data is from yfinance format
    """
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, data_info = load_and_prepare_data(
        data_path, batch_size=batch_size, use_yfinance=use_yfinance
    )

    config = DiscountModelConfig(
        input_dim=data_info["num_features"],
        hidden_dims=[128, 64, 32],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiscountDetector(config).to(device)

    class_weights = torch.tensor(data_info["class_weights"], dtype=torch.float32).to(
        device
    )

    trainer = DiscountTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        class_weights=class_weights,
    )

    logger.info("Starting training...")
    logger.info(f"Class weights (applied): {class_weights.tolist()}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        confidence_threshold=confidence_threshold,
    )

    final_metrics = results["final_metrics"]

    output = {
        "training_date": datetime.utcnow().isoformat(),
        "data_info": data_info,
        "final_metrics": final_metrics,
        "confidence_threshold": confidence_threshold,
    }

    model_path = os.path.join(output_dir, "discount_model.pt")
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)
    logger.info(f"Model saved to {model_path}")

    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    print("\n" + "=" * 60)
    print("DISCOUNT DETECTION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training samples: {data_info['train_size']}")
    print(f"Validation samples: {data_info['val_size']}")
    print(f"Precision: {final_metrics['precision']:.1%}")
    print(f"Recall: {final_metrics['recall']:.1%}")
    print(f"F1 Score: {final_metrics['f1']:.1%}")
    print(f"Accuracy: {final_metrics['accuracy']:.1%}")
    print(f"Avg Confidence: {final_metrics['avg_confidence']:.1%}")
    print(f"Model saved: {model_path}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train discount detection model")
    parser.add_argument(
        "--data", type=str, default="data/mnq_historical.csv", help="Path to OHLCV data"
    )
    parser.add_argument(
        "--output", type=str, default="results/discount", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.80, help="Confidence threshold"
    )
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="Data is from yfinance format (lowercase columns)",
    )

    args = parser.parse_args()

    results = train_discount_model(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        patience=args.patience,
        confidence_threshold=args.confidence,
        use_yfinance=args.yfinance,
    )
