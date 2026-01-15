"""
Discount Detector Model

Simple MLP binary classifier for detecting successful discount opportunities.
Predicts YES (1) or NO (0) based on whether price will reach 3:1 target.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiscountModelConfig:
    """Configuration for discount detector model"""

    input_dim: int = 14
    hidden_dims: List[int] = None
    output_dim: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class DiscountDetector(nn.Module):
    """
    Simple MLP for discount detection.

    Architecture:
    - Input: 14 price-based features
    - Hidden1: 128 + LayerNorm + GELU + Dropout
    - Hidden2: 64 + LayerNorm + GELU + Dropout
    - Hidden3: 32 + LayerNorm + GELU
    - Output: 2 (NO/YES probabilities)
    """

    def __init__(self, config: Optional[DiscountModelConfig] = None):
        super().__init__()

        if config is None:
            config = DiscountModelConfig()

        self.config = config

        if config.hidden_dims is None:
            config.hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, config.output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Dict with logits and probabilities
        """
        hidden = self.network(x)
        logits = self.output(hidden)

        probs = F.softmax(logits, dim=-1)

        confidence = probs[:, 1]  # P(YES)

        return {
            "logits": logits,
            "probabilities": probs,
            "confidence": confidence,
        }

    def predict(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.80,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference prediction.

        Args:
            x: Input features
            confidence_threshold: Minimum confidence for YES

        Returns:
            Dict with predictions
        """
        with torch.no_grad():
            output = self(x)

            predictions = (output["confidence"] >= confidence_threshold).long()
            predictions = predictions.where(
                output["confidence"] >= confidence_threshold,
                torch.zeros_like(predictions),
            )

            return {
                "predictions": predictions,
                "confidence": output["confidence"],
                "probabilities": output["probabilities"],
            }


class DiscountPredictor:
    """
    High-level predictor class for discount detection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[DiscountModelConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.config = config or DiscountModelConfig()
        self.model = DiscountDetector(self.config).to(device)

        if model_path is not None:
            self.load(model_path)

        self.feature_names = [
            "fib_distance_pts",
            "fib_distance_pct",
            "price_vs_fib",
            "is_discount",
            "momentum_3d",
            "momentum_1d",
            "rsi_14",
            "volume_ratio",
            "atr",
            "volatility",
            "hour_of_day",
            "day_of_week",
            "distance_to_weekly_high",
            "distance_to_weekly_low",
        ]

    def predict_single(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Predict on single sample.

        Args:
            features: Feature vector (14 features)
            confidence_threshold: Min confidence for YES

        Returns:
            Dict with prediction results
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)

        confidence = output["confidence"].item()
        prob_yes = output["probabilities"][0, 1].item()
        prob_no = output["probabilities"][0, 0].item()

        if confidence >= confidence_threshold:
            signal = "YES"
        else:
            signal = "NO"

        return {
            "signal": signal,
            "confidence": confidence,
            "prob_yes": prob_yes,
            "prob_no": prob_no,
            "is_discount": features[3] > 0.5,
        }

    def predict_batch(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.80,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch.

        Args:
            features: Feature matrix (n_samples, 14)
            confidence_threshold: Min confidence for YES

        Returns:
            Dict with batch predictions
        """
        x = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(x)

        confidence = output["confidence"]
        predictions = (confidence >= confidence_threshold).long()

        return {
            "predictions": predictions,
            "confidence": confidence,
            "probabilities": output["probabilities"],
        }

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        try:
            self.model = torch.jit.load(path, map_location=self.device)
            logger.info(f"TorchScript model loaded from {path}")
        except Exception:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Checkpoint loaded from {path}")


def create_model(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[DiscountDetector, torch.optim.Optimizer]:
    """
    Create model and optimizer.

    Args:
        config_path: Path to config file
        checkpoint_path: Optional checkpoint to load
        device: Device to use

    Returns:
        Tuple of (model, optimizer)
    """
    config = DiscountModelConfig()
    model = DiscountDetector(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model, optimizer


if __name__ == "__main__":
    model, optimizer = create_model()

    model.eval()

    batch_size = 4
    features = np.random.randn(batch_size, 14).astype(np.float32)

    x = torch.tensor(features).to(model.device)

    with torch.no_grad():
        output = model(x)

    print("=== Discount Detector Model Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(
        f"Confidence range: {output['confidence'].min():.3f} - {output['confidence'].max():.3f}"
    )

    predictions = model.predict(x, confidence_threshold=0.80)
    print(f"\nPredictions: {predictions['predictions'].cpu().numpy()}")
    print(f"Confidences: {predictions['confidence'].cpu().numpy()}")
